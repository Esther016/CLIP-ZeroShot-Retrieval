import os
import json
import random
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from transformers import CLIPModel, CLIPProcessor

# ================= config =================
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR if (_SCRIPT_DIR / "data").exists() else _SCRIPT_DIR.parent

DATA_DIR = str(PROJECT_ROOT / "data")
ANN_FILE = os.path.join(DATA_DIR, "annotations", "captions_val2017.json")
IMG_DIR = os.path.join(DATA_DIR, "val2017")

CACHE_DIR = str(PROJECT_ROOT / "outputs" / "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

NUM_IMAGES = 5000
CAPTIONS_PER_IMAGE = 5
SEED = 42

MODEL_NAME = "openai/clip-vit-base-patch32"
MODEL_TAG = MODEL_NAME.replace("/", "_")

CACHE_IMG = os.path.join(CACHE_DIR, f"img_{NUM_IMAGES}_{MODEL_TAG}.pt")
CACHE_TXT = os.path.join(CACHE_DIR, f"txt_{NUM_IMAGES}_{MODEL_TAG}.pt")
CACHE_META = os.path.join(CACHE_DIR, f"meta_{NUM_IMAGES}_{MODEL_TAG}.json")
# =======================================


def enable_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_rank_indices(sim: torch.Tensor) -> torch.Tensor:
    """
    Stable descending ranking with deterministic tie-break by smaller image index.
    """
    n_img = sim.shape[1]
    idx = torch.arange(n_img, device=sim.device, dtype=sim.dtype)
    adjusted = sim - idx.unsqueeze(0) * 1e-12
    return torch.argsort(adjusted, dim=1, descending=True)


def main():
    enable_determinism(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    coco = COCO(ANN_FILE)
    img_ids_all = sorted(coco.getImgIds())
    sampled_img_ids = random.sample(img_ids_all, k=NUM_IMAGES)

    captions = []
    caption_ids = []       # ★ Freeze caption order
    gt_img_index = []

    print("Building subset...")
    for img_idx, img_id in enumerate(sampled_img_ids):
        ann_ids = sorted(coco.getAnnIds(imgIds=img_id))
        anns = coco.loadAnns(ann_ids)
        for a in anns[:CAPTIONS_PER_IMAGE]:
            captions.append(a["caption"])
            caption_ids.append(a["id"])
            gt_img_index.append(img_idx)

    print(f"Subset built: images={len(sampled_img_ids)}, captions={len(captions)}")

    # ========== Load CLIP ==========
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    # ========== Image Embeddings ==========
    if os.path.exists(CACHE_IMG):
        print("Loading cached image embeddings...")
        img_emb = torch.load(CACHE_IMG, map_location=device)
    else:
        print("Encoding images...")
        image_embs = []
        for img_id in tqdm(sampled_img_ids, desc="Encoding images"):
            info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(IMG_DIR, info["file_name"])
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            image_embs.append(emb.squeeze(0))

        img_emb = torch.stack(image_embs)
        torch.save(img_emb, CACHE_IMG)

    # ========== Text Embeddings ==========
    if os.path.exists(CACHE_TXT):
        print("Loading cached text embeddings...")
        txt_emb = torch.load(CACHE_TXT, map_location=device)
    else:
        print("Encoding texts...")
        text_embs = []
        for text in tqdm(captions, desc="Encoding texts"):
            inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                emb = model.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            text_embs.append(emb.squeeze(0))

        txt_emb = torch.stack(text_embs)
        torch.save(txt_emb, CACHE_TXT)

    # ========== Save META  ==========
    meta = {
        "num_images": NUM_IMAGES,
        "captions_per_image": CAPTIONS_PER_IMAGE,
        "seed": SEED,
        "sampled_img_ids": sampled_img_ids,
        "caption_ids": caption_ids,
        "gt_img_index": gt_img_index
    }
    with open(CACHE_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved embeddings and meta.")

    # ========== Evaluate Recall ==========
    sim = txt_emb @ img_emb.T
    ranks = stable_rank_indices(sim)

    def recall_at_k(k):
        correct = 0
        for i in range(len(gt_img_index)):
            if gt_img_index[i] in ranks[i, :k]:
                correct += 1
        return correct / len(gt_img_index)

    print("-" * 48)
    print(f"R@1  = {recall_at_k(1):.4f}")
    print(f"R@5  = {recall_at_k(5):.4f}")
    print(f"R@10 = {recall_at_k(10):.4f}")
    print("-" * 48)
    print("Baseline done.")

    # ========== Save Captions ==========
    captions_file = os.path.join(CACHE_DIR, f"captions_{NUM_IMAGES}_{MODEL_TAG}.json")
    with open(captions_file, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False)
    print("Saved captions ->", captions_file)


if __name__ == "__main__":
    main()

