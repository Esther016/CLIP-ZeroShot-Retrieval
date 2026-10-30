"""
Control experiments for prompt ensembling.

Stage 1 (this script, needs the CLIP model + cached embeddings):
  For EVERY cached caption (not just annotated failures), apply every template
  set and every pooling mode, and record the rank of the ground-truth image.

  Template sets (built with the same functions as improve_subset.py):
    generic           build_templates(c)                        K=5
    Object, Attribute, Action, Spatial, Count, Context
                      build_templates_by_category(c, [cat])     K=10
    Object+Attribute  build_templates_by_category(c, [Object, Attribute])  K=15

  Because every set is applied to every caption, the oracle setting
  (templates matched to the human label) and the placebo settings
  (generic / wrong-category templates) are just different columns of the output.

Output: outputs/controls/ranks.csv.gz with one row per caption:
  idx, gt_img_index, rank_baseline, rank_<set>__<pool>, ...
  (rank 0 = retrieved at top-1; hit@K <=> rank < K)

Then run src/analyze_controls.py (CPU only, no model needed).
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from improve_subset import build_templates, build_templates_by_category, enable_determinism


_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR if (_SCRIPT_DIR / "data").exists() else _SCRIPT_DIR.parent

CATEGORIES = ["Object", "Attribute", "Action", "Spatial", "Count", "Context"]
TEMPLATE_SETS = {
    "generic": lambda c: build_templates(c),
    **{cat: (lambda c, cat=cat: build_templates_by_category(c, [cat])) for cat in CATEGORIES},
    "Object+Attribute": lambda c: build_templates_by_category(c, ["Object", "Attribute"]),
}
# (name, mode, tau). lse_t1 is the setting used in the original report; with cosine
# similarities in a narrow range, tau=1 makes logsumexp rank almost exactly like mean.
# lse_t0.01 is included to show the interpolation towards max.
POOLINGS = [("mean", "mean", None), ("max", "max", None),
            ("lse_t1", "logsumexp", 1.0), ("lse_t0.01", "logsumexp", 0.01)]


def gt_rank(sim: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    sim: [B, N_img], gt: [B]. Returns the 0-based rank of the ground-truth image,
    using the same tie-break as stable_rank_indices (smaller image index first).
    """
    n_img = sim.shape[1]
    idx = torch.arange(n_img, device=sim.device, dtype=sim.dtype)
    adjusted = sim - idx.unsqueeze(0) * 1e-12
    gt_score = adjusted.gather(1, gt.unsqueeze(1))
    return (adjusted > gt_score).sum(dim=1)


def pool(block: torch.Tensor, mode: str, tau) -> torch.Tensor:
    """block: [K, N_img] -> [N_img]"""
    if mode == "mean":
        return block.mean(dim=0)
    if mode == "max":
        return block.max(dim=0).values
    return tau * torch.logsumexp(block / tau, dim=0)


@torch.no_grad()
def encode(model, processor, texts, device, batch_size):
    out = []
    for i in range(0, len(texts), batch_size):
        inputs = processor(text=texts[i:i + batch_size], return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to(device)
        emb = model.get_text_features(**inputs)
        if not torch.is_tensor(emb):  # transformers>=5 returns an output object
            emb = emb.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.float())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_images", type=int, default=5000)
    ap.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--cache_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "cache"))
    ap.add_argument("--cache_tag", type=str, default="",
                    help="Model tag in cache file names (default: derived from --model_name)")
    ap.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "controls"))
    ap.add_argument("--captions_per_chunk", type=int, default=64,
                    help="Captions processed together; lower it if you run out of (GPU) memory")
    ap.add_argument("--text_batch_size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N captions (debugging)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    enable_determinism(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_tag = args.cache_tag or args.model_name.replace("/", "_")
    paths = {k: os.path.join(args.cache_dir, f"{k}_{args.num_images}_{model_tag}.{ext}")
             for k, ext in [("img", "pt"), ("txt", "pt"), ("meta", "json"), ("captions", "json")]}
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing cache file: {p} (run src/main.py first)")

    img_emb = torch.load(paths["img"], map_location="cpu").float().to(device)
    txt_emb = torch.load(paths["txt"], map_location="cpu").float().to(device)
    with open(paths["meta"], encoding="utf-8") as f:
        gt_all = torch.tensor(json.load(f)["gt_img_index"], dtype=torch.long)
    with open(paths["captions"], encoding="utf-8") as f:
        captions = json.load(f)

    n = args.limit or len(captions)
    print(f"Device: {device} | captions: {n} | images: {img_emb.shape[0]}")

    model = CLIPModel.from_pretrained(args.model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_name)

    cols = {"idx": np.arange(n), "gt_img_index": gt_all[:n].numpy()}
    cols["rank_baseline"] = gt_rank(txt_emb[:n] @ img_emb.T, gt_all[:n].to(device)).cpu().numpy()
    results = {f"rank_{s}__{p}": np.empty(n, dtype=np.int32) for s in TEMPLATE_SETS for p, _, _ in POOLINGS}
    n_templates = {}

    for start in tqdm(range(0, n, args.captions_per_chunk), desc="Caption chunks"):
        stop = min(start + args.captions_per_chunk, n)

        # Unique texts for this chunk + per (caption, set) index lists into them.
        texts, text_pos, plan = [], {}, []
        for i in range(start, stop):
            per_set = {}
            for s, fn in TEMPLATE_SETS.items():
                ids = []
                for t in fn(captions[i]):
                    if t not in text_pos:
                        text_pos[t] = len(texts)
                        texts.append(t)
                    ids.append(text_pos[t])
                per_set[s] = ids
                n_templates.setdefault(s, len(ids))
            plan.append(per_set)

        sims = encode(model, processor, texts, device, args.text_batch_size) @ img_emb.T  # [T, N_img]

        for s in TEMPLATE_SETS:
            for p, mode, tau in POOLINGS:
                pooled = torch.stack([pool(sims[plan[j][s]], mode, tau) for j in range(stop - start)])
                results[f"rank_{s}__{p}"][start:stop] = gt_rank(pooled, gt_all[start:stop].to(device)).cpu().numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "ranks.csv.gz")
    pd.DataFrame({**cols, **results}).to_csv(out_csv, index=False)
    with open(os.path.join(args.out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"model_name": args.model_name, "num_captions": n, "seed": args.seed,
                   "templates_per_set": n_templates,
                   "poolings": [{"name": p, "mode": m, "tau": t} for p, m, t in POOLINGS]}, f, indent=2)

    base_hit = cols["rank_baseline"] < np.array([[1], [5], [10]])
    print("Baseline R@1/5/10 = " + " / ".join(f"{h.mean():.4f}" for h in base_hit))
    print(f"[OK] Saved -> {out_csv}")


if __name__ == "__main__":
    main()
