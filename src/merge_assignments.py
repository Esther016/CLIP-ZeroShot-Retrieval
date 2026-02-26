from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV or Excel, preserving empty strings."""
    if path.suffix.lower() in [".xlsx", ".xls"]:
        # Read first sheet by default
        return pd.read_excel(path)
    return pd.read_csv(path)


def merge_assignments(
    project_root: Path,
    input_dir: str = "failure_analysis",
    files: list[str] | None = None,
    out_raw: str = "assign_all_raw.csv",
    out_dedup: str = "assign_all.csv",
) -> None:
    """
    Merge assignment files into one CSV.

    - Reads available files (CSV/XLSX) under input_dir.
    - Writes assign_all_raw.csv (no dedup).
    - Writes assign_all.csv (dedup if possible).
    """
    if files is None:
        files = [
            "assign_A.csv",
            "assign_B.csv",
            "assign_C.csv",
            "assign_overlap.csv",
            "assign_overlap.xlsx"
        ]

    indir = project_root / input_dir
    if not indir.exists():
        raise FileNotFoundError(f"Input directory not found: {indir}")

    dfs: list[pd.DataFrame] = []
    used_files: list[str] = []

    print(f"[INFO] Working directory: {project_root}")
    print(f"[INFO] Input directory: {indir}")

    for fn in files:
        p = indir / fn
        if p.exists():
            df = _read_any(p)
            # keep track of source file
            df["__source_file"] = fn
            dfs.append(df)
            used_files.append(fn)
            print(f"[FOUND] {fn}  rows={len(df)}  cols={len(df.columns)}")
        else:
            print(f"[MISS ] {fn}")

    if not dfs:
        raise RuntimeError(
            "No assignment files found. "
            "Make sure assign_A/B/C/overlap files exist under failure_analysis/."
        )

    merged = pd.concat(dfs, ignore_index=True)

    # Save raw merged (no dedup)
    raw_path = indir / out_raw
    merged.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved raw merged -> {raw_path}  rows={len(merged)}")

    # Decide dedup keys (only if present)
    candidate_keys = [
        ["ann_id"],                      # best if exists
        ["query_idx", "img_idx"],         # common pair
        ["query_idx", "gt_img_idx"],      # sometimes used
        ["caption_id"],                   # sometimes used
    ]

    dedup_keys: list[str] = []
    for keys in candidate_keys:
        if all(k in merged.columns for k in keys):
            dedup_keys = keys
            break

    if dedup_keys:
        before = len(merged)
        deduped = merged.drop_duplicates(subset=dedup_keys, keep="first").copy()
        after = len(deduped)
        print(f"[INFO] Dedup keys: {dedup_keys}  removed={before - after}")
    else:
        # No safe key: do NOT dedup automatically
        deduped = merged
        print("[WARN] No dedup keys found (ann_id / query_idx+img_idx / etc.). "
              "Skipping dedup to avoid corrupting data.")

    out_path = indir / out_dedup
    deduped.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved final merged -> {out_path}  rows={len(deduped)}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    root = script_dir if (script_dir / "failure_analysis").exists() else script_dir.parent
    merge_assignments(root)
