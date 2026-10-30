"""
Stage 2 of the control experiments (CPU only): turn outputs/controls/ranks.csv.gz
into tables. All deltas are paired (same captions, same images) and come with
95% bootstrap CIs over captions.

  A. Full set (all captions): does prompt ensembling help overall, and how many
     captions flip hit->miss vs miss->hit?
  B. By baseline outcome: R@1 successes vs all R@1 failures vs annotated failures.
     Gains that only show up on selected failures are a selection effect.
  C. Annotated failures, matched vs mismatched templates: rows = human category,
     columns = template set. If the matched (diagonal) cell is not better than
     generic / wrong-category templates, "category-aware" is not what helps.
  D. Reproduction check against the original report (rows NOT deduplicated).

Usage:
  python src/analyze_controls.py
  python src/analyze_controls.py --pooling max --k 10
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR if (_SCRIPT_DIR / "data").exists() else _SCRIPT_DIR.parent

KS = [1, 5, 10]
ROW_CATEGORIES = ["Object", "Attribute", "Action", "Context", "Count", "Ambiguous"]
# Original report numbers (rows not deduplicated): (subset, template set) -> pooling -> (R@5, R@10) in %
REPORTED = {
    ("Object+Attribute", "Object+Attribute"): {"max": (40.23, 52.87), "mean": (43.68, 52.87), "lse_t1": (43.68, 52.87)},
    ("Object", "Object"): {"max": (37.10, 53.23), "mean": (37.10, 54.84), "lse_t1": (38.71, 54.84)},
    ("Attribute", "Attribute"): {"max": (44.00, 52.00), "mean": (48.00, 52.00), "lse_t1": (48.00, 52.00)},
    ("Action", "Action"): {"max": (37.50, 66.67), "mean": (33.33, 66.67), "lse_t1": (33.33, 66.67)},
}


def boot_ci(x: np.ndarray, B: int, seed: int, chunk: int = 100):
    """Mean of x with a percentile bootstrap 95% CI (resampling rows of x)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    means = []
    for b in range(0, B, chunk):
        idx = rng.integers(0, n, size=(min(chunk, B - b), n))
        means.append(x[idx].mean(axis=1))
    lo, hi = np.percentile(np.concatenate(means), [2.5, 97.5])
    return x.mean(), lo, hi


def fmt_pp(m, lo, hi):
    return f"{m * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks_csv", type=str, default=str(PROJECT_ROOT / "outputs" / "controls" / "ranks.csv.gz"))
    ap.add_argument("--annotations_csv", type=str,
                    default=str(PROJECT_ROOT / "failure_analysis" / "analysis_all" / "annotations_clean.csv"))
    ap.add_argument("--pooling", type=str, default="mean", help="Pooling used for tables B and C")
    ap.add_argument("--k", type=int, default=5, choices=KS, help="Recall@K used for table C")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.ranks_csv)
    df = pd.read_csv(args.ranks_csv)
    ann = pd.read_csv(args.annotations_csv)
    cat_col = "category_clean" if "category_clean" in ann.columns else "category"

    sets = sorted({c[len("rank_"):].split("__")[0] for c in df.columns if "__" in c})
    pools = sorted({c.split("__")[1] for c in df.columns if "__" in c})
    base = df["rank_baseline"].to_numpy()
    hit = lambda r, k: (r < k).astype(np.int8)
    lines = [f"# Control experiments\n",
             f"Captions: {len(df)} | template sets: {', '.join(sets)} | poolings: {', '.join(pools)}",
             f"Deltas in percentage points with 95% paired bootstrap CI (B={args.B}, seed={args.seed}).\n"]

    # ---------------- A. Full set ----------------
    rows = []
    for s in sets:
        for p in pools:
            r = df[f"rank_{s}__{p}"].to_numpy()
            row = {"template_set": s, "pooling": p}
            for k in KS:
                d = hit(r, k) - hit(base, k)
                m, lo, hi = boot_ci(d.astype(float), args.B, args.seed)
                row[f"R@{k}"] = round(hit(r, k).mean() * 100, 2)
                row[f"dR@{k}"] = fmt_pp(m, lo, hi)
                row[f"gained@{k}"] = int((d == 1).sum())
                row[f"lost@{k}"] = int((d == -1).sum())
            rows.append(row)
    tab_a = pd.DataFrame(rows)
    tab_a.to_csv(os.path.join(out_dir, "A_full_set.csv"), index=False)
    lines += ["## A. Full set (all captions)\n",
              "Baseline R@1/5/10 = " + " / ".join(f"{hit(base, k).mean() * 100:.2f}" for k in KS) + "\n",
              tab_a.to_markdown(index=False), ""]

    # ---------------- B. By baseline outcome ----------------
    ann_idx = set(ann["idx"].astype(int))
    strata = {
        "R@1 successes": base < 1,
        "all R@1 failures": base >= 1,
        "R@1 failures, not annotated": (base >= 1) & ~df["idx"].isin(ann_idx).to_numpy(),
        "annotated failures (unique)": df["idx"].isin(ann_idx).to_numpy(),
    }
    rows = []
    for s in sets:
        r = df[f"rank_{s}__{args.pooling}"].to_numpy()
        for name, mask in strata.items():
            row = {"template_set": s, "stratum": name, "n": int(mask.sum())}
            for k in KS:
                d = (hit(r[mask], k) - hit(base[mask], k)).astype(float)
                row[f"dR@{k}"] = fmt_pp(*boot_ci(d, args.B, args.seed))
            rows.append(row)
    tab_b = pd.DataFrame(rows)
    tab_b.to_csv(os.path.join(out_dir, "B_by_baseline_outcome.csv"), index=False)
    lines += [f"## B. By baseline outcome (pooling = {args.pooling})\n", tab_b.to_markdown(index=False), ""]

    # ---------------- C. Matched vs mismatched (annotated, deduplicated) ----------------
    # Deduplicate on (idx, category): a caption the two annotators labeled the same way counts once;
    # a caption they disagreed on counts once under each label.
    ann_u = ann.drop_duplicates(subset=["idx", cat_col])
    k = args.k
    by_idx = df.set_index("idx")
    rows, gap_rows = [], []
    row_defs = [(c, [c]) for c in ROW_CATEGORIES] + [("Object+Attribute", ["Object", "Attribute"])]
    for row_name, cats in row_defs:
        idx = ann_u.loc[ann_u[cat_col].isin(cats), "idx"].astype(int).to_numpy()
        if len(idx) == 0:
            continue
        b = hit(by_idx.loc[idx, "rank_baseline"].to_numpy(), k)
        row = {"annotated_category": row_name, "n": len(idx), f"baseline R@{k}": round(b.mean() * 100, 2)}
        deltas = {}
        for s in sets:
            deltas[s] = (hit(by_idx.loc[idx, f"rank_{s}__{args.pooling}"].to_numpy(), k) - b).astype(float)
            m, lo, hi = boot_ci(deltas[s], args.B, args.seed)
            row[s] = fmt_pp(m, lo, hi) + (" *" if s == row_name else "")
        rows.append(row)
        if row_name in sets:
            mismatched = [s for s in sets if s not in (row_name, "generic")]
            gap_rows.append({
                "annotated_category": row_name, "n": len(idx),
                "matched - generic": fmt_pp(*boot_ci(deltas[row_name] - deltas["generic"], args.B, args.seed)),
                "matched - mean(wrong category)": fmt_pp(*boot_ci(
                    deltas[row_name] - np.mean([deltas[s] for s in mismatched], axis=0), args.B, args.seed)),
            })
    tab_c = pd.DataFrame(rows)
    tab_c.to_csv(os.path.join(out_dir, f"C_matched_vs_mismatched_R{k}_{args.pooling}.csv"), index=False)
    tab_gap = pd.DataFrame(gap_rows)
    tab_gap.to_csv(os.path.join(out_dir, f"C_gap_R{k}_{args.pooling}.csv"), index=False)
    lines += [f"## C. Annotated failures: ΔR@{k} by template set (pooling = {args.pooling})\n",
              "`*` marks the template set that matches the human label (the original 'category-aware' setting).\n",
              tab_c.to_markdown(index=False), "",
              "Matched-template advantage (paired, same captions):\n", tab_gap.to_markdown(index=False), ""]

    # ---------------- D. Reproduction check ----------------
    rows = []
    for (subset, s), by_pool in REPORTED.items():
        cats = subset.split("+")
        idx = ann.loc[ann[cat_col].isin(cats), "idx"].astype(int).to_numpy()  # NOT deduplicated, as in the report
        for p, (r5, r10) in by_pool.items():
            col = f"rank_{s}__{p}"
            if col not in by_idx.columns:
                continue
            r = by_idx.loc[idx, col].to_numpy()
            rows.append({"subset": subset, "n_rows": len(idx), "pooling": p,
                         "R@5 now": round(hit(r, 5).mean() * 100, 2), "R@5 reported": r5,
                         "R@10 now": round(hit(r, 10).mean() * 100, 2), "R@10 reported": r10})
    tab_d = pd.DataFrame(rows)
    tab_d.to_csv(os.path.join(out_dir, "D_reproduction_check.csv"), index=False)
    lines += ["## D. Reproduction check vs. original report (rows not deduplicated)\n",
              tab_d.to_markdown(index=False), ""]

    report = os.path.join(out_dir, f"controls_report_{args.pooling}_R{k}.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"[OK] Wrote {report}")


if __name__ == "__main__":
    main()
