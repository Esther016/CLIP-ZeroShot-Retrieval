import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) Normalization helpers
# =========================

CANON_CATS = ["Ambiguous", "Attribute", "Object", "Action", "Spatial", "Context", "Count"]

def normalize_category(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s_low = s.lower()

    # common typos / variants
    mapping = {
        "ambiguou": "Ambiguous",
        "ambiguous": "Ambiguous",
        "attribute": "Attribute",
        "attr": "Attribute",
        "object": "Object",
        "obj": "Object",
        "action": "Action",
        "interaction": "Action",
        "spatial": "Spatial",
        "space": "Spatial",
        "context": "Context",
        "scene": "Context",
        "count": "Count",
        "counting": "Count",
        "plurality": "Count",
    }

    # remove non-letters for robustness
    s_key = re.sub(r"[^a-z]", "", s_low)
    if s_key in mapping:
        return mapping[s_key]

    # fallback: Title-case then try partial match
    s_title = s.strip().title()
    for c in CANON_CATS:
        if c.lower() in s_low:
            return c
    return s_title  # last resort


def normalize_subtype(x: str) -> str:
    """Only meaningful when category == Ambiguous."""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if not s:
        return ""

    # remove non-letters
    s_key = re.sub(r"[^a-z]", "", s)

    mapping = {
        "nearduplicate": "nearduplicate",
        "nearduplicate": "nearduplicate",
        "dup": "nearduplicate",
        "duplicate": "nearduplicate",
        "underspecified": "underspecified",
        "underspec": "underspecified",
        "underspecification": "underspecified",
        "vague": "underspecified",
    }
    if s_key in mapping:
        return mapping[s_key]

    # heuristic contains
    if "duplicate" in s_key or "dup" in s_key:
        return "nearduplicate"
    if "under" in s_key or "vague" in s_key or "spec" in s_key:
        return "underspecified"
    return s  # keep raw if unknown


# =========================
# 2) Analysis + plotting
# =========================

def plot_bar(series_counts: pd.Series, title: str, xlabel: str, ylabel: str, save_path: str):
    labels = series_counts.index.tolist()
    values = series_counts.values.astype(float)

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to annotation CSV (e.g., assign_overlap.csv)")
    parser.add_argument("--outdir", default="analysis_out", help="Output folder for plots/tables")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # --------- Required columns check ----------
    required = {"idx", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # --------- Clean fields ----------
    df["category_clean"] = df["category"].apply(normalize_category)

    if "ambiguous_subtype" in df.columns:
        df["ambiguous_subtype_clean"] = df["ambiguous_subtype"].apply(normalize_subtype)
    else:
        df["ambiguous_subtype_clean"] = ""

    # If category is not Ambiguous, subtype should be empty
    df.loc[df["category_clean"] != "Ambiguous", "ambiguous_subtype_clean"] = ""

    # Save cleaned file
    cleaned_path = os.path.join(args.outdir, "annotations_clean.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"[OK] Saved cleaned annotations -> {cleaned_path}")

    # --------- Overall distribution (counts + %) ----------
    total_n = len(df)
    counts = df["category_clean"].value_counts().reindex(CANON_CATS).fillna(0).astype(int)
    perc = (counts / total_n * 100.0).round(2)

    summary = pd.DataFrame({"count": counts, "percent": perc})
    summary_path = os.path.join(args.outdir, "category_distribution_overall.csv")
    summary.to_csv(summary_path)
    print(f"[OK] Saved overall distribution -> {summary_path}")
    print("\nOverall category distribution:")
    print(summary)

    # Plot overall %
    plot_bar(
        perc,
        title=f"Overall Failure Category Distribution (N={total_n})",
        xlabel="Category",
        ylabel="Percentage (%)",
        save_path=os.path.join(args.outdir, "fig_overall_category_percent.png"),
    )
    print(f"[OK] Saved plot -> fig_overall_category_percent.png")

    # --------- Ambiguous subtype breakdown ----------
    amb = df[df["category_clean"] == "Ambiguous"]
    if len(amb) > 0:
        amb_counts = amb["ambiguous_subtype_clean"].value_counts()
        # keep only the two canonical ones in a fixed order
        amb_counts = amb_counts.reindex(["nearduplicate", "underspecified"]).fillna(0).astype(int)
        amb_perc = (amb_counts / len(amb) * 100.0).round(2)

        amb_summary = pd.DataFrame({"count": amb_counts, "percent_within_ambiguous": amb_perc})
        amb_path = os.path.join(args.outdir, "ambiguous_subtype_distribution.csv")
        amb_summary.to_csv(amb_path)
        print(f"[OK] Saved ambiguous subtype distribution -> {amb_path}")
        print("\nAmbiguous subtype distribution:")
        print(amb_summary)

        plot_bar(
            amb_perc,
            title=f"Ambiguous Subtype Breakdown (Ambiguous n={len(amb)})",
            xlabel="Ambiguous subtype",
            ylabel="Percentage within Ambiguous (%)",
            save_path=os.path.join(args.outdir, "fig_ambiguous_subtype_percent.png"),
        )
        print(f"[OK] Saved plot -> fig_ambiguous_subtype_percent.png")
    else:
        print("[INFO] No Ambiguous cases found; skipping ambiguous subtype plot.")

    # --------- Actionable-only distribution (exclude Ambiguous) ----------
    actionable = df[df["category_clean"] != "Ambiguous"]
    actionable_n = len(actionable)
    if actionable_n > 0:
        act_cats = ["Attribute", "Object", "Action", "Spatial", "Context", "Count"]
        act_counts = actionable["category_clean"].value_counts().reindex(act_cats).fillna(0).astype(int)
        act_perc = (act_counts / actionable_n * 100.0).round(2)

        act_summary = pd.DataFrame({"count": act_counts, "percent_within_actionable": act_perc})
        act_path = os.path.join(args.outdir, "category_distribution_actionable_only.csv")
        act_summary.to_csv(act_path)
        print(f"[OK] Saved actionable-only distribution -> {act_path}")
        print("\nActionable-only distribution:")
        print(act_summary)

        plot_bar(
            act_perc,
            title=f"Actionable-only Category Distribution (n={actionable_n})",
            xlabel="Category (excluding Ambiguous)",
            ylabel="Percentage (%)",
            save_path=os.path.join(args.outdir, "fig_actionable_category_percent.png"),
        )
        print(f"[OK] Saved plot -> fig_actionable_category_percent.png")
    else:
        print("[INFO] All cases are Ambiguous; skipping actionable-only plot.")

    # --------- Optional: agreement if overlap columns exist ----------
    # If later you have two annotators in the SAME file, e.g. category_A, category_B
    if {"category_A", "category_B"}.issubset(df.columns):
        from sklearn.metrics import cohen_kappa_score

        a = df["category_A"].apply(normalize_category)
        b = df["category_B"].apply(normalize_category)

        agree = (a == b).mean()
        kappa = cohen_kappa_score(a, b)

        with open(os.path.join(args.outdir, "agreement.txt"), "w", encoding="utf-8") as f:
            f.write(f"Percent agreement: {agree:.4f}\n")
            f.write(f"Cohen's kappa: {kappa:.4f}\n")

        print("\n[OK] Agreement computed:")
        print(f"Percent agreement: {agree:.4f}")
        print(f"Cohen's kappa:     {kappa:.4f}")
        print("[OK] Saved -> agreement.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
