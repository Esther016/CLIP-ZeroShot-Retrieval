import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

# ============== 配置 ==============
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR if (_SCRIPT_DIR / "data").exists() else _SCRIPT_DIR.parent

CACHE_DIR = str(PROJECT_ROOT / "outputs" / "cache")
NUM_IMAGES = 5000
MODEL_TAG = "openai_clip-vit-base-patch32".replace("/", "_")

CACHE_IMG = os.path.join(CACHE_DIR, f"img_{NUM_IMAGES}_{MODEL_TAG}.pt")
CACHE_TXT = os.path.join(CACHE_DIR, f"txt_{NUM_IMAGES}_{MODEL_TAG}.pt")
CACHE_META = os.path.join(CACHE_DIR, f"meta_{NUM_IMAGES}_{MODEL_TAG}.json")

DATA_DIR = str(PROJECT_ROOT / "data")
ANN_FILE = os.path.join(DATA_DIR, "annotations", "captions_val2017.json")
IMG_DIR = os.path.join(DATA_DIR, "val2017")

OUTPUT_DIR = str(PROJECT_ROOT / "failure_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 新增：任务分配配置
OVERLAP_N = 30
PER_PERSON_N = 50
TEAM = ["A", "B", "C"]
# =================================


def visualize_failure(caption, gt_path, pred_path, save_path):
    gt = Image.open(gt_path).convert("RGB").resize((300, 300))
    pred = Image.open(pred_path).convert("RGB").resize((300, 300))

    canvas = Image.new("RGB", (650, 380), (255, 255, 255))
    canvas.paste(gt, (20, 50))
    canvas.paste(pred, (330, 50))

    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), caption[:80], fill=(0, 0, 0))
    draw.text((20, 360), "Ground Truth", fill=(0, 128, 0))
    draw.text((330, 360), "Top-1 Retrieved", fill=(255, 0, 0))

    canvas.save(save_path)


def write_assignment(path, idx_list, captions, gt_img_index, top1_idx):
    """生成任务分配CSV文件，包含标注列"""
    with open(path, "w", encoding="utf-8") as f:
        # 新增category和ambiguous_subtype列
        f.write("idx,caption,gt_img_idx,pred_img_idx,category,ambiguous_subtype\n")
        for i in idx_list:
            cap = captions[i].replace('"', "'")
            # 修复：tensor转int避免索引问题
            pred_idx = int(top1_idx[i])
            gt_idx = int(gt_img_index[i])
            f.write(f'{i},"{cap}",{gt_idx},{pred_idx},,\n')


def save_visuals(tag, idx_list, captions, gt_img_index, top1_idx, image_paths, output_dir):
    """按任务组保存可视化图片"""
    vis_dir = os.path.join(output_dir, f"vis_{tag}")
    os.makedirs(vis_dir, exist_ok=True)
    
    for i in tqdm(idx_list, desc=f"Saving visuals ({tag})"):
        gt_path = image_paths[int(gt_img_index[i])]
        pred_path = image_paths[int(top1_idx[i])]  # 修复：tensor转int
        save_path = os.path.join(vis_dir, f"fail_{i}.jpg")
        visualize_failure(captions[i], gt_path, pred_path, save_path)


def main():
    print("Loading cache...")
    img_emb = torch.load(CACHE_IMG, map_location="cpu")
    txt_emb = torch.load(CACHE_TXT, map_location="cpu")

    with open(CACHE_META, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sampled_img_ids = meta["sampled_img_ids"]
    caption_ids = meta["caption_ids"]
    gt_img_index = meta["gt_img_index"]

    coco = COCO(ANN_FILE)

    # Build image paths (order fixed)
    image_paths = []
    for img_id in sampled_img_ids:
        info = coco.loadImgs(img_id)[0]
        image_paths.append(os.path.join(IMG_DIR, info["file_name"]))

    # Build captions (order fixed)
    captions = []
    for cid in caption_ids:
        ann = coco.loadAnns([cid])[0]
        captions.append(ann["caption"])

    # Retrieval
    sim = txt_emb @ img_emb.T
    top1_idx = torch.argmax(sim, dim=1)

    failures = []
    for i in range(len(captions)):
        # 修复：tensor转int比较
        if int(top1_idx[i]) != int(gt_img_index[i]):
            failures.append(i)

    print(f"Found {len(failures)} R@1 failures.")

    # Sample 200 failures for annotation
    random.seed(42)
    sampled = random.sample(failures, 200)

    # ========== 新增：任务分配逻辑 ==========
    # 划分任务集
    overlap = sampled[:OVERLAP_N]
    A_set = sampled[OVERLAP_N:OVERLAP_N + PER_PERSON_N]
    B_set = sampled[OVERLAP_N + PER_PERSON_N:OVERLAP_N + 2*PER_PERSON_N]
    C_set = sampled[OVERLAP_N + 2*PER_PERSON_N:OVERLAP_N + 3*PER_PERSON_N]
    backup = sampled[OVERLAP_N + 3*PER_PERSON_N:]

    # 保存原始总CSV（保留原有逻辑）
    csv_path = os.path.join(OUTPUT_DIR, "failure_samples.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("idx,caption,gt_img_idx,pred_img_idx\n")
        for i in sampled:
            cap = captions[i].replace('"', "'")
            f.write(f'{i},"{cap}",{int(gt_img_index[i])},{int(top1_idx[i])}\n')  # 修复：tensor转int
    print(f"Saved original CSV: {csv_path}")

    # 保存任务分配CSV
    write_assignment(os.path.join(OUTPUT_DIR, "assign_overlap.csv"), overlap, captions, gt_img_index, top1_idx)
    write_assignment(os.path.join(OUTPUT_DIR, "assign_A.csv"), A_set, captions, gt_img_index, top1_idx)
    write_assignment(os.path.join(OUTPUT_DIR, "assign_B.csv"), B_set, captions, gt_img_index, top1_idx)
    write_assignment(os.path.join(OUTPUT_DIR, "assign_C.csv"), C_set, captions, gt_img_index, top1_idx)
    write_assignment(os.path.join(OUTPUT_DIR, "assign_backup.csv"), backup, captions, gt_img_index, top1_idx)
    print("Saved assignment CSV files for team members.")

    # ========== 可视化：按任务组独立保存 ==========
    save_visuals("overlap", overlap, captions, gt_img_index, top1_idx, image_paths, OUTPUT_DIR)
    save_visuals("A", A_set, captions, gt_img_index, top1_idx, image_paths, OUTPUT_DIR)
    save_visuals("B", B_set, captions, gt_img_index, top1_idx, image_paths, OUTPUT_DIR)
    save_visuals("C", C_set, captions, gt_img_index, top1_idx, image_paths, OUTPUT_DIR)
    save_visuals("backup", backup, captions, gt_img_index, top1_idx, image_paths, OUTPUT_DIR)

    print("All failure visualizations saved by assignment group.")


if __name__ == "__main__":
    main()