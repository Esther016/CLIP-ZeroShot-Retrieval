import json
import glob
import os
from pathlib import Path
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR if (_SCRIPT_DIR / "data").exists() else _SCRIPT_DIR.parent

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # Only take the structured result files
    subset_results_dir = PROJECT_ROOT / "outputs" / "subset_results"
    files = sorted(glob.glob(str(subset_results_dir / "subset_results_*.json")))
    if not files:
        raise SystemExit(f"No files matched subset_results_*.json in {subset_results_dir}.")

    rows = []
    for fp in files:
        d = load_json(fp)

        cats = d.get("categories", [])
        cat_tag = ",".join(cats) if cats else "ALL"

        baseline = d.get("baseline", {})
        improved = d.get("improved", {})
        delta = d.get("delta_pct_points", {})

        rows.append({
            "file": os.path.basename(fp),
            "subset_size": d.get("subset_size"),
            "categories": cat_tag,
            "pooling": d.get("pooling"),
            "tau": d.get("tau"),
            "K_templates": d.get("templates_per_caption"),
            "seed": d.get("seed"),

            "baseline_R@1": baseline.get("R@1"),
            "baseline_R@5": baseline.get("R@5"),
            "baseline_R@10": baseline.get("R@10"),

            "improved_R@1": improved.get("R@1"),
            "improved_R@5": improved.get("R@5"),
            "improved_R@10": improved.get("R@10"),

            "delta_pp_R@1": delta.get("R@1"),
            "delta_pp_R@5": delta.get("R@5"),
            "delta_pp_R@10": delta.get("R@10"),
        })

    df = pd.DataFrame(rows)

    # Make it easier to read
    df = df.sort_values(by=["categories", "subset_size", "pooling"]).reset_index(drop=True)

    # Convert rates to percent for readability (keep delta already in percentage points)
    for c in ["baseline_R@1","baseline_R@5","baseline_R@10","improved_R@1","improved_R@5","improved_R@10"]:
        df[c] = df[c].apply(lambda x: None if x is None else round(x * 100, 2))
    for c in ["delta_pp_R@1","delta_pp_R@5","delta_pp_R@10"]:
        df[c] = df[c].apply(lambda x: None if x is None else round(x, 2))

    out_summary_dir = PROJECT_ROOT / "outputs" / "summary"
    out_summary_dir.mkdir(parents=True, exist_ok=True)
    out_csv = str(out_summary_dir / "summary_subset_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Wrote {out_csv}")
    print("\n=== Summary (percent; delta in pp) ===")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 140):
        print(df)

if __name__ == "__main__":
    main()
