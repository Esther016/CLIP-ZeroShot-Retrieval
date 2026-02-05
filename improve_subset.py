import os
import json
import argparse
import random
import torch
import pandas as pd
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ---------------------------
# Helper: Recall@K on subset
# ---------------------------
def recall_at_k_on_subset(sim_matrix, gt_img_index, subset_indices, k: int) -> float:
    """
    sim_matrix: [N_text, N_img]
    gt_img_index: list[int] length N_text
    subset_indices: list[int] indices into text dimension
    """
    sim_sub = sim_matrix[subset_indices]  # [n_sub, N_img]
    ranks = torch.argsort(sim_sub, dim=1, descending=True)[:, :k]  # [n_sub, k]

    gt = torch.tensor([gt_img_index[i] for i in subset_indices], dtype=torch.long)
    hit = (ranks == gt.unsqueeze(1)).any(dim=1).float().mean().item()
    return hit


# ---------------------------
# Helper: Build templated texts
# ---------------------------
def build_templates(caption: str):
    """
    Prompt ensembling templates. Keep it small (3-6) for speed.
    These are intentionally more "attribute/object" focused than "a photo of {caption}".
    """
    c = caption.strip()
    return [
        c,
        f"a photo of {c}",
        f"an image of {c}",
        f"{c}. close-up view.",
        f"describing the image: {c}",
    ]


def encode_texts(model, processor, texts, device, batch_size=64):
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding templated texts"):
        batch = texts[i:i+batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_emb.append(emb.detach().cpu())
    return torch.cat(all_emb, dim=0)  # [N, D]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=5000)
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--cache_dir", type=str, default="cache")

    # Choose one:
    parser.add_argument("--annotations_csv", type=str, default="", help="annotations_clean.csv with columns [idx, category] (preferred)")
    parser.add_argument("--failure_samples_csv", type=str, default="", help="failure_samples.csv with column [idx] (fallback)")

    # subset selection
    parser.add_argument("--include_categories", type=str, default="Attribute,Object",
                        help="Comma-separated categories for subset, e.g. Attribute,Object")
    parser.add_argument("--max_subset", type=int, default=999999, help="Cap subset size for quick tests")
    parser.add_argument("--seed", type=int, default=42)

    # prompt ensembling controls
    parser.add_argument("--do_prompt_ensemble", action="store_true", help="If set, apply prompt ensembling to subset")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_tag = args.model_name.replace("/", "_")
    cache_img = os.path.join(args.cache_dir, f"img_{args.num_images}_{model_tag}.pt")
    cache_txt = os.path.join(args.cache_dir, f"txt_{args.num_images}_{model_tag}.pt")
    cache_meta = os.path.join(args.cache_dir, f"meta_{args.num_images}_{model_tag}.json")

    if not (os.path.exists(cache_img) and os.path.exists(cache_txt) and os.path.exists(cache_meta)):
        raise FileNotFoundError(
            "Missing cache files. Expected:\n"
            f"- {cache_img}\n- {cache_txt}\n- {cache_meta}\n"
            "Run main.py first to generate them."
        )

    print("Loading cached embeddings/meta...")
    img_emb = torch.load(cache_img, map_location="cpu")   # [N_img, D]
    txt_emb = torch.load(cache_txt, map_location="cpu")   # [N_txt, D]

    with open(cache_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    captions_per_image = meta.get("captions_per_image", 5)
    caption_ids = meta.get("caption_ids", None)  # optional
    gt_img_index = meta["gt_img_index"]          # length N_txt

    # ---------
    # Build subset indices
    # ---------
    subset_indices = None

    if args.annotations_csv:
        df = pd.read_csv(args.annotations_csv)
        if "idx" not in df.columns or "category" not in df.columns:
            raise ValueError("annotations_csv must include columns: idx, category")

        include = [x.strip() for x in args.include_categories.split(",") if x.strip()]
        df_sub = df[df["category"].isin(include)].copy()

        subset_indices = df_sub["idx"].astype(int).tolist()

        print(f"Subset from annotations_csv: categories={include}, n={len(subset_indices)}")

    elif args.failure_samples_csv:
        df = pd.read_csv(args.failure_samples_csv)
        if "idx" not in df.columns:
            raise ValueError("failure_samples_csv must include column: idx")
        subset_indices = df["idx"].astype(int).tolist()
        print(f"Subset from failure_samples_csv: n={len(subset_indices)}")

    else:
        raise ValueError("Provide either --annotations_csv or --failure_samples_csv.")

    # cap subset
    if len(subset_indices) > args.max_subset:
        random.shuffle(subset_indices)
        subset_indices = subset_indices[:args.max_subset]
        print(f"Capped subset to n={len(subset_indices)}")

    # ---------
    # Baseline evaluation on subset
    # ---------
    sim_baseline = txt_emb @ img_emb.T  # [N_txt, N_img]

    r1_b = recall_at_k_on_subset(sim_baseline, gt_img_index, subset_indices, 1)
    r5_b = recall_at_k_on_subset(sim_baseline, gt_img_index, subset_indices, 5)
    r10_b = recall_at_k_on_subset(sim_baseline, gt_img_index, subset_indices, 10)

    print("\n=== BASELINE (subset) ===")
    print(f"Subset size: {len(subset_indices)}")
    print(f"R@1  = {r1_b*100:.2f}%")
    print(f"R@5  = {r5_b*100:.2f}%")
    print(f"R@10 = {r10_b*100:.2f}%")

    # Optional: overall baseline (for context only)
    # overall_r1 = (torch.argmax(sim_baseline, dim=1) == torch.tensor(gt_img_index)).float().mean().item()
    # print(f"[Context] Overall R@1 = {overall_r1*100:.2f}%")

    # ---------
    # Prompt Ensembling (only on subset)
    # ---------
    if not args.do_prompt_ensemble:
        print("\n[INFO] --do_prompt_ensemble not set. Done.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nLoading CLIP for prompt ensembling:", args.model_name, "| device:", device)
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    # Build templated text list for subset in a flat batch:
    # For each subset idx, create K templates; later max-pool similarity across templates.
    template_lists = []
    flat_texts = []
    for idx in subset_indices:
        cap = meta.get("captions", None)
        # We do NOT rely on meta["captions"] (often not stored). We reconstruct caption text from meta if you store it.
        # If you don't store captions in meta, easiest is: keep captions list in main.py or save captions to disk.
        # Here we assume you can read captions from a file `captions_{num_images}_{model_tag}.json` if needed.
        # ---- Practical fallback: store captions in meta in your main.py (recommended).
        pass

    # Practical approach (recommended):
    # You already have txt_emb but not raw captions in cache. So we need captions text for templating.
    # The simplest is: in main.py, save captions list to a json file.
    captions_file = os.path.join(args.cache_dir, f"captions_{args.num_images}_{model_tag}.json")
    if not os.path.exists(captions_file):
        raise FileNotFoundError(
            f"Need captions text to do prompt ensembling, but not found:\n{captions_file}\n\n"
            "Fix (one-time): in main.py, save captions list to this file.\n"
            "Example:\n"
            "with open(captions_file, 'w', encoding='utf-8') as f: json.dump(captions, f)\n"
        )

    with open(captions_file, "r", encoding="utf-8") as f:
        captions = json.load(f)

    # Build flattened templates
    for idx in subset_indices:
        c = captions[idx]
        tpls = build_templates(c)
        template_lists.append(tpls)
        flat_texts.extend(tpls)

    K = len(template_lists[0])
    print(f"Templates per caption: K={K}")
    print(f"Total templated texts to encode: {len(flat_texts)}")

    flat_emb = encode_texts(model, processor, flat_texts, device=device, batch_size=args.batch_size)  # [n_sub*K, D]
    flat_emb = flat_emb.to(torch.float32)

    # Compute similarity for subset templates vs all images
    img_emb_f32 = img_emb.to(torch.float32)
    sim_templates = flat_emb @ img_emb_f32.T  # [n_sub*K, N_img]

    # Max-pool over templates for each caption in subset
    sim_improved_sub = []
    for i in range(len(subset_indices)):
        block = sim_templates[i*K:(i+1)*K]  # [K, N_img]
        sim_improved_sub.append(block.max(dim=0).values.unsqueeze(0))
    sim_improved_sub = torch.cat(sim_improved_sub, dim=0)  # [n_sub, N_img]

    # Evaluate improved subset recall
    # We need a sim_matrix shaped [N_txt, N_img] to reuse the recall fn; easiest: directly compute on subset
    ranks = torch.argsort(sim_improved_sub, dim=1, descending=True)

    gt = torch.tensor([gt_img_index[i] for i in subset_indices], dtype=torch.long)

    r1_i = (ranks[:, :1] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
    r5_i = (ranks[:, :5] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
    r10_i = (ranks[:, :10] == gt.unsqueeze(1)).any(dim=1).float().mean().item()

    print("\n=== IMPROVED (subset, prompt ensembling) ===")
    print(f"R@1  = {r1_i*100:.2f}%   (Δ = {(r1_i-r1_b)*100:.2f}%)")
    print(f"R@5  = {r5_i*100:.2f}%   (Δ = {(r5_i-r5_b)*100:.2f}%)")
    print(f"R@10 = {r10_i*100:.2f}%  (Δ = {(r10_i-r10_b)*100:.2f}%)")

    # Save summary
    out = {
        "subset_size": len(subset_indices),
        "categories": args.include_categories,
        "baseline": {"R@1": r1_b, "R@5": r5_b, "R@10": r10_b},
        "improved": {"R@1": r1_i, "R@5": r5_i, "R@10": r10_i},
        "delta_pct_points": {
            "R@1": (r1_i - r1_b) * 100.0,
            "R@5": (r5_i - r5_b) * 100.0,
            "R@10": (r10_i - r10_b) * 100.0,
        }
    }
    out_path = "subset_improvement_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Saved results -> {out_path}")


if __name__ == "__main__":
    main()
