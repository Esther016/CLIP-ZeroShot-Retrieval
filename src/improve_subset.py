import os
import json
import argparse
import random
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR if (_SCRIPT_DIR / "data").exists() else _SCRIPT_DIR.parent

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
    """
    c = caption.strip()
    return [
        c,
        f"a photo of {c}",
        f"an image of {c}",
        f"{c}. close-up view.",
        f"describing the image: {c}",
    ]


def build_templates_by_category(caption: str, cats: list[str]) -> list[str]:
    """
    Category-aware prompt templates.
    cats: list of category names, e.g. ["Object"], ["Attribute"], ["Object","Attribute"]
    """
    c = (caption or "").strip()
    if not c:
        return [""]

    base = [
        c,
        f"a photo of {c}",
        f"an image of {c}",
        f"describing the image: {c}",
        f"{c}. close-up view.",
    ]

    obj = [
        f"a photo of a {c}",
        f"the main object is {c}",
        f"there is {c} in the scene",
        f"{c} on a table",
        f"{c} in the background",
    ]
    attr = [
        f"{c} with distinctive color and texture",
        f"{c} showing clear attributes",
        f"{c} with visible details",
        f"{c} in a specific style",
        f"{c} with notable appearance",
    ]
    act = [
        f"a person is {c}",
        f"someone is {c}",
        f"an action: {c}",
        f"{c} happening in the scene",
        f"{c} in progress",
    ]
    spatial = [
        f"{c} on the left side",
        f"{c} on the right side",
        f"{c} in the center",
        f"{c} in the foreground",
        f"{c} in the background",
    ]
    count = [
        f"multiple {c}",
        f"two {c}",
        f"three {c}",
        f"many {c}",
        f"several {c}",
    ]
    context = [
        f"{c} indoors",
        f"{c} outdoors",
        f"{c} in a kitchen",
        f"{c} in a street scene",
        f"{c} in a natural environment",
    ]

    cat_map = {
        "Object": obj,
        "Attribute": attr,
        "Action": act,
        "Spatial": spatial,
        "Count": count,
        "Context": context,
    }

    extra: list[str] = []
    for cat in cats or []:
        cat = (cat or "").strip()
        if cat in cat_map:
            extra.extend(cat_map[cat])

    # De-duplicate while preserving order
    out = []
    seen = set()
    for t in base + extra:
        if t not in seen:
            out.append(t)
            seen.add(t)

    return out


def encode_texts(model, processor, texts, device, batch_size=64):
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding templated texts"):
        batch = texts[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_emb.append(emb.detach().cpu())
    return torch.cat(all_emb, dim=0)  # [N, D]


def pool_templates(block: torch.Tensor, mode: str = "max", tau: float = 1.0) -> torch.Tensor:
    """
    block: [K, N_img] similarities for K templates of one caption.
    returns: [N_img] pooled similarities
    mode:
      - max: max over K
      - mean: average over K
      - logsumexp: tau * logsumexp(block / tau) over K  (softmax-like)
    """
    mode = mode.lower().strip()
    if mode == "max":
        return block.max(dim=0).values
    if mode == "mean":
        return block.mean(dim=0)
    if mode == "logsumexp":
        # numerically stable logsumexp pooling
        return tau * torch.logsumexp(block / tau, dim=0)
    raise ValueError(f"Unknown pooling mode: {mode}. Use one of: max, mean, logsumexp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=5000)
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--cache_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "cache"))

    # Choose one:
    parser.add_argument("--annotations_csv", type=str, default="",
                        help="annotations_clean.csv with columns [idx, category] (preferred)")
    parser.add_argument("--failure_samples_csv", type=str, default="",
                        help="failure_samples.csv with column [idx] (fallback)")

    # subset selection
    parser.add_argument("--include_categories", type=str, default="Attribute,Object",
                        help="Comma-separated categories for subset, e.g. Attribute,Object")
    parser.add_argument("--max_subset", type=int, default=999999, help="Cap subset size for quick tests")
    parser.add_argument("--seed", type=int, default=42)

    # prompt ensembling controls
    parser.add_argument("--do_prompt_ensemble", action="store_true",
                        help="If set, apply prompt ensembling to subset")
    parser.add_argument("--batch_size", type=int, default=64)

    # NEW: pooling over templates
    parser.add_argument("--pooling", type=str, default="max",
                        choices=["max", "mean", "logsumexp"],
                        help="How to pool similarities across templates (default: max)")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Temperature for logsumexp pooling (only used if --pooling logsumexp)")
    parser.add_argument("--save_hits_csv", action="store_true",
                    help="If set, save per-sample hit@K (baseline vs improved) for bootstrapping.")


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
            "Run main-v2.py first to generate them."
        )

    print("Loading cached embeddings/meta...")
    img_emb = torch.load(cache_img, map_location="cpu")  # [N_img, D]
    txt_emb = torch.load(cache_txt, map_location="cpu")  # [N_txt, D]

    with open(cache_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    gt_img_index = meta["gt_img_index"]  # length N_txt

    # ---------
    # Build subset indices
    # ---------
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

    captions_file = os.path.join(args.cache_dir, f"captions_{args.num_images}_{model_tag}.json")
    if not os.path.exists(captions_file):
        raise FileNotFoundError(
            f"Need captions text to do prompt ensembling, but not found:\n{captions_file}\n\n"
            "Fix (one-time): in main-v2.py, save captions list to this file.\n"
        )

    with open(captions_file, "r", encoding="utf-8") as f:
        captions = json.load(f)

    use_cats = [x.strip() for x in args.include_categories.split(",") if x.strip()]

    template_lists = []
    flat_texts = []
    for idx in subset_indices:
        c = captions[idx]
        tpls = build_templates_by_category(c, use_cats)
        template_lists.append(tpls)
        flat_texts.extend(tpls)

    K = len(template_lists[0])
    print(f"Templates per caption: K={K}")
    print(f"Total templated texts to encode: {len(flat_texts)}")
    print(f"Pooling mode: {args.pooling}" + (f" (tau={args.tau})" if args.pooling == "logsumexp" else ""))

    flat_emb = encode_texts(model, processor, flat_texts, device=device, batch_size=args.batch_size)  # [n_sub*K, D]
    flat_emb = flat_emb.to(torch.float32)

    img_emb_f32 = img_emb.to(torch.float32)
    sim_templates = flat_emb @ img_emb_f32.T  # [n_sub*K, N_img]

    # Pool over templates for each caption
    sim_improved_sub = []
    for i in range(len(subset_indices)):
        block = sim_templates[i * K:(i + 1) * K]  # [K, N_img]
        pooled = pool_templates(block, mode=args.pooling, tau=args.tau)  # [N_img]
        sim_improved_sub.append(pooled.unsqueeze(0))
    sim_improved_sub = torch.cat(sim_improved_sub, dim=0)  # [n_sub, N_img]

    # Evaluate improved subset recall
    ranks = torch.argsort(sim_improved_sub, dim=1, descending=True)
    gt = torch.tensor([gt_img_index[i] for i in subset_indices], dtype=torch.long)

    r1_i = (ranks[:, :1] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
    r5_i = (ranks[:, :5] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
    r10_i = (ranks[:, :10] == gt.unsqueeze(1)).any(dim=1).float().mean().item()

    print("\n=== IMPROVED (subset, prompt ensembling) ===")
    print(f"R@1  = {r1_i*100:.2f}%   (Δ = {(r1_i-r1_b)*100:.2f}%)")
    print(f"R@5  = {r5_i*100:.2f}%   (Δ = {(r5_i-r5_b)*100:.2f}%)")
    print(f"R@10 = {r10_i*100:.2f}%  (Δ = {(r10_i-r10_b)*100:.2f}%)")

    # --- per-sample hits (for bootstrapping) ---
    baseline_sub = sim_baseline[subset_indices]  # [n_sub, N_img]
    base_ranks = torch.argsort(baseline_sub, dim=1, descending=True)

    def hit_at_k(ranks_mat, gt_vec, k):
        return (ranks_mat[:, :k] == gt_vec.unsqueeze(1)).any(dim=1).to(torch.int32).cpu().numpy()

    base_hit1 = hit_at_k(base_ranks, gt, 1)
    base_hit5 = hit_at_k(base_ranks, gt, 5)
    base_hit10 = hit_at_k(base_ranks, gt, 10)

    imp_hit1 = hit_at_k(ranks, gt, 1)
    imp_hit5 = hit_at_k(ranks, gt, 5)
    imp_hit10 = hit_at_k(ranks, gt, 10)

    if args.save_hits_csv:
        import numpy as np
        cats_tag = "-".join(use_cats) if use_cats else "ALL"
        out_hits_dir = PROJECT_ROOT / "outputs" / "subset_hits"
        os.makedirs(out_hits_dir, exist_ok=True)
        out_hits = str(out_hits_dir / f"subset_hits_{cats_tag}_n{len(subset_indices)}_{args.pooling}_seed{args.seed}.csv")
        hits_df = pd.DataFrame({
            "idx": subset_indices,
            "gt_img_index": gt.cpu().numpy(),
            "baseline_hit@1": base_hit1,
            "baseline_hit@5": base_hit5,
            "baseline_hit@10": base_hit10,
            "improved_hit@1": imp_hit1,
            "improved_hit@5": imp_hit5,
            "improved_hit@10": imp_hit10,
        })
        hits_df.to_csv(out_hits, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved per-sample hits -> {out_hits}")


    out = {
        "subset_size": len(subset_indices),
        "categories": use_cats,
        "seed": args.seed,
        "pooling": args.pooling,
        "tau": args.tau if args.pooling == "logsumexp" else None,
        "templates_per_caption": K,
        "baseline": {"R@1": r1_b, "R@5": r5_b, "R@10": r10_b},
        "improved": {"R@1": r1_i, "R@5": r5_i, "R@10": r10_i},
        "delta_pct_points": {
            "R@1": (r1_i - r1_b) * 100.0,
            "R@5": (r5_i - r5_b) * 100.0,
            "R@10": (r10_i - r10_b) * 100.0,
        }
    }

    # Unique output filename (avoid overwriting)
    cat_tag = "-".join(use_cats) if use_cats else "ALL"
    out_results_dir = PROJECT_ROOT / "outputs" / "subset_results"
    os.makedirs(out_results_dir, exist_ok=True)
    out_path = str(out_results_dir / f"subset_results_{cat_tag}_n{len(subset_indices)}_{args.pooling}_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Saved results -> {out_path}")


if __name__ == "__main__":
    main()
