import argparse
import numpy as np
import pandas as pd

def bootstrap_delta(hit_base, hit_imp, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(hit_base)
    deltas = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)  # resample with replacement
        deltas[b] = hit_imp[idx].mean() - hit_base[idx].mean()
    mean = deltas.mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_pos = (deltas > 0).mean()
    return mean, lo, hi, p_pos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits_csv", required=True, help="subset_hits_*.csv produced by improve_subset.py --save_hits_csv")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.hits_csv)

    print(f"\n=== Bootstrapping on {args.hits_csv} | B={args.B} seed={args.seed} ===")
    for k in [1, 5, 10]:
        b = df[f"baseline_hit@{k}"].to_numpy()
        i = df[f"improved_hit@{k}"].to_numpy()
        mean, lo, hi, p_pos = bootstrap_delta(b, i, B=args.B, seed=args.seed)
        print(f"ΔR@{k}: mean={mean*100:.2f}pp, 95% CI=[{lo*100:.2f}, {hi*100:.2f}]pp, P(Δ>0)={p_pos:.3f}")

if __name__ == "__main__":
    main()
