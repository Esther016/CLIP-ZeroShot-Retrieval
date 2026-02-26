import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration and color palette
# ==========================================
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR if (_SCRIPT_DIR / "data").exists() else _SCRIPT_DIR.parent

# Input files
SUMMARY_CSV = str(PROJECT_ROOT / "outputs" / "summary" / "summary_subset_results.csv")
ASSIGN_CSV  = str(PROJECT_ROOT / "failure_analysis" / "analysis_all" / "annotations_clean.csv")

# Bootstrap data files
HITS_FILES = {
    "max": str(PROJECT_ROOT / "outputs" / "subset_hits" / "subset_hits_Action_n25_max_seed42.csv"),
    "mean": str(PROJECT_ROOT / "outputs" / "subset_hits" / "subset_hits_Action_n25_mean_seed42.csv"),
    "logsumexp": str(PROJECT_ROOT / "outputs" / "subset_hits" / "subset_hits_Action_n25_logsumexp_seed42.csv"),
}

OUTDIR = str(PROJECT_ROOT / "outputs" / "figures")
os.makedirs(OUTDIR, exist_ok=True)

# --- Core styling: custom color palette (Tableau-like) ---
# A more stable and professional palette than default RGB
COLORS = {
    "max":       "#4E79A7",  # Deep Blue
    "mean":      "#F28E2B",  # Muted Orange
    "logsumexp": "#59A14F",  # Forest Green
}
COLOR_GRAY = "#595959"       # Text/border gray

# --- Global plotting style ---
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans'] # Prefer Arial
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = COLOR_GRAY
plt.rcParams['xtick.color'] = COLOR_GRAY
plt.rcParams['ytick.color'] = COLOR_GRAY
plt.rcParams['text.color'] = 'black'
plt.rcParams['figure.dpi'] = 150 

# ==========================================
# 2. Data loading and preprocessing
# ==========================================
def generate_dummy_data():
    """Generate dummy data for testing if CSV is missing."""
    print("[INFO] CSV not found. Using dummy data for demonstration.")
    data = {
        "categories": ["Object+Attribute", "Object+Attribute", "Object+Attribute",
                       "Object", "Object", "Object",
                       "Attribute", "Attribute", "Attribute",
                       "Action", "Action", "Action"],
        "pooling": ["max", "mean", "logsumexp"] * 4,
        "delta_pp_R@5":  [3.45, 6.90, 6.90, 3.23, 3.23, 4.84, 0.0, 4.0, 4.0, 4.17, 0.0, 0.0],
        "delta_pp_R@10": [1.15, 1.15, 1.15, 1.61, 3.23, 3.23, 0.0, 0.0, 0.0, 20.83, 20.83, 20.83]
    }
    return pd.DataFrame(data)

# Try loading CSV; fall back to dummy data if missing
if os.path.exists(SUMMARY_CSV):
    df = pd.read_csv(SUMMARY_CSV)
else:
    df = generate_dummy_data()

# Standardize category names and ordering
subset_order = ["Object+Attribute", "Object", "Attribute", "Action"]
pool_order   = ["max", "mean", "logsumexp"]

# Normalize category separators: "Object,Attribute" -> "Object+Attribute"
# This is a common data-cleaning step
df["categories"] = df["categories"].str.replace(",", "+")

# ==========================================
# 3. Plotting functions (core styling logic)
# ==========================================

def setup_axis(ax, xlabel, ylabel):
    """Apply common axis styling."""
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add light gray grid behind bars
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)
    
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=8, color='#333333')
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=8, color='#333333')
def grouped_bar_beautiful(metric, fname, ylabel):
    """Draw grouped bars with value labels (legend location adjusted only)."""
    # Prepare data
    cats = [c for c in subset_order if c in df["categories"].unique()]
    x = np.arange(len(cats))
    width = 0.25  # Bar width
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot each pooling strategy
    for i, p in enumerate(pool_order):
        vals = []
        for cat in cats:
            row = df[(df["categories"] == cat) & (df["pooling"] == p)]
            if len(row) == 0:
                vals.append(0.0)
            else:
                vals.append(float(row.iloc[0][metric]))
        
        # Compute x positions
        x_pos = x + (i - 1) * width
        
        # Draw bars (color/style unchanged)
        bars = ax.bar(
            x_pos, vals, width, 
            label=p, 
            color=COLORS[p], 
            edgecolor='white',  
            linewidth=0.7,
            zorder=3            
        )
        
        # Value labels (unchanged)
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            y_offset = 0.2 if height < 0.5 else height + 0.2
            
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                y_offset, 
                f"{val:.1f}", 
                ha='center', va='bottom', 
                fontsize=9, color='#333333', fontweight='bold'
            )

    # Axis settings (unchanged)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=0)
    setup_axis(ax, "Failure Subset", ylabel)
    
    # --- Dynamic legend placement ---
    # Compare total values between left and right subsets
    left_sum = df[df["categories"].isin(subset_order[:2])][metric].sum()  # Left: Object+Attribute/Object
    right_sum = df[df["categories"].isin(subset_order[2:])][metric].sum() # Right: Attribute/Action
    # Higher left -> upper right, higher right -> upper left
    legend_loc = "upper right" if left_sum > right_sum else "upper left"
    
    ax.legend(title="Pooling Strategy", frameon=False, loc=legend_loc)

    # Save outputs (unchanged)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname + ".png"), dpi=300)
    plt.savefig(os.path.join(OUTDIR, fname + ".pdf"))
    plt.close()
    print(f"[Output] Saved {fname} (legend at {legend_loc})")

# ==========================================
# 4. Generate figures
# ==========================================

# --- Fig 1 & 2: Grouped bars (R@5 and R@10) ---
grouped_bar_beautiful("delta_pp_R@5",  "fig_pooling_ablation_delta_r5_clean",  "Recall@5 Gain (pp)")
grouped_bar_beautiful("delta_pp_R@10", "fig_pooling_ablation_delta_r10_clean", "Recall@10 Gain (pp)")

# --- Fig 3: Best Pooling per Subset ---
# Select best strategy
best = (
    df.sort_values(by="delta_pp_R@10", ascending=False)
      .drop_duplicates(subset=["categories"], keep="first")
)
# Enforce fixed order
best = best.set_index("categories").reindex(subset_order).reset_index()

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(best))
# Assign colors by best pooling strategy
bar_colors = [COLORS[p] for p in best["pooling"]]

bars = ax.bar(
    x, best["delta_pp_R@10"], 
    width=0.6, 
    color=bar_colors,
    edgecolor='white',
    zorder=3
)

# Annotate values
for bar, val, pooling in zip(bars, best["delta_pp_R@10"], best["pooling"]):
    height = bar.get_height()
    y_offset = 0.2 if height < 0.5 else height + 0.2
    
    # Numeric label
    ax.text(
        bar.get_x() + bar.get_width()/2, y_offset, 
        f"+{val:.1f}", 
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )
    # Put pooling label inside bar if space allows
    if height > 2:
        ax.text(
            bar.get_x() + bar.get_width()/2, height/2, 
            pooling, 
            ha='center', va='center', color='white', fontsize=9, fontweight='bold'
        )
    else:
        # If not enough space, put it above the bar
        ax.text(
            bar.get_x() + bar.get_width()/2, y_offset + 1.5, 
            f"({pooling})", 
            ha='center', va='bottom', color=COLOR_GRAY, fontsize=8
        )

ax.set_xticks(x)
ax.set_xticklabels(best["categories"])
setup_axis(ax, "Failure Subset", "Best Recall@10 Gain (pp)")

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_best_delta_r10_clean.png"), dpi=300)
plt.savefig(os.path.join(OUTDIR, "fig_best_delta_r10_clean.pdf"))
plt.close()
print(f"[Output] Saved fig_best_delta_r10_clean")

# --- Fig 4: Bootstrap (if files exist) ---
# Includes a fallback-style flow for easier preview
def plot_bootstrap_beautiful():
    B = 2000
    rng = np.random.default_rng(42)
    delta_samples = {}

    # Read actual files + bootstrap ΔR@10
    for pooling, path in HITS_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing hits file: {path}")
            continue

        hits = pd.read_csv(path)

        # Expected columns: baseline_hit@10 / improved_hit@10
        base_col = "baseline_hit@10"
        imp_col  = "improved_hit@10"
        if base_col not in hits.columns or imp_col not in hits.columns:
            print(f"[WARN] Unexpected columns in {path}: {list(hits.columns)}")
            continue

        base = hits[base_col].to_numpy(dtype=float)
        imp  = hits[imp_col].to_numpy(dtype=float)
        n = len(base)

        deltas = np.empty(B, dtype=float)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            deltas[b] = 100.0 * (imp[idx].mean() - base[idx].mean())

        delta_samples[pooling] = deltas
        print(f"[INFO] Bootstrapped {pooling}: n={n}, B={B}")

    # Exit early if still empty (avoid KeyError)
    if len(delta_samples) == 0:
        print("[WARN] No bootstrap samples computed. Check HITS_FILES paths.")
        return

    # Plot only available pooling strategies (keep order)
    keys = [p for p in pool_order if p in delta_samples]
    data_to_plot = [delta_samples[p] for p in keys]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Boxplot styling
    boxprops = dict(linewidth=1.5, color=COLOR_GRAY)
    medianprops = dict(linewidth=2, color='#D32F2F')
    whiskerprops = dict(linewidth=1.5, color=COLOR_GRAY)
    capprops = dict(linewidth=1.5, color=COLOR_GRAY)

    bp = ax.boxplot(
        data_to_plot,
        tick_labels=keys,        # Matplotlib 3.9+ uses tick_labels
        patch_artist=True,
        showfliers=False,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        widths=0.5,
        zorder=3
    )

    for patch, p in zip(bp["boxes"], keys):
        patch.set_facecolor(COLORS[p])
        patch.set_alpha(0.6)
        patch.set_edgecolor(COLORS[p])

    setup_axis(ax, "Pooling Strategy", "Bootstrap Δ Recall@10 (pp)")
    ax.set_title("Robustness Analysis (Action Subset)", fontsize=12, pad=10)

    # Optional: draw zero reference line
    ax.axhline(0, linestyle="--", linewidth=1, color=COLOR_GRAY, alpha=0.6, zorder=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_bootstrap_action_delta_r10_clean.png"), dpi=300)
    plt.savefig(os.path.join(OUTDIR, "fig_bootstrap_action_delta_r10_clean.pdf"))
    plt.close()
    print(f"[Output] Saved fig_bootstrap_action_delta_r10_clean")

plot_bootstrap_beautiful()

# ==========================================
# 5. Overlap agreement (category kappa + confusion heatmap)
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Replace with your own paths/params (consistent with your script)
OVERLAP_A_CSV = str(PROJECT_ROOT / "failure_analysis" / "assign_overlap1.csv")  # Annotator A
OVERLAP_B_CSV = str(PROJECT_ROOT / "failure_analysis" / "assign_overlap2.csv")  # Annotator B
OUTDIR = str(PROJECT_ROOT / "outputs" / "figures")
COLOR_GRAY = "#7f7f7f"  # Keep consistent with existing style

def _clean_cat(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x == "" or x.lower() in ["nan", "none"]:
        return None
    return x

def plot_overlap_agreement_and_confusion():
    if not (os.path.exists(OVERLAP_A_CSV) and os.path.exists(OVERLAP_B_CSV)):
        print(f"[WARN] Missing overlap csvs: {OVERLAP_A_CSV}, {OVERLAP_B_CSV}")
        return

    A = pd.read_csv(OVERLAP_A_CSV)
    B = pd.read_csv(OVERLAP_B_CSV)

    # ---- robust merge key: prefer idx if exists, else fallback to gt_img_index, else row order
    key = None
    for cand in ["idx", "gt_img_index", "image_id", "ann_id"]:
        if cand in A.columns and cand in B.columns:
            key = cand
            break
    if key is None:
        A = A.reset_index().rename(columns={"index": "_row"})
        B = B.reset_index().rename(columns={"index": "_row"})
        key = "_row"

    # ---- category column name
    cat_col = None
    for cand in ["category", "Category", "label", "failure_category"]:
        if cand in A.columns:
            cat_col = cand
            break
    if cat_col is None or cat_col not in B.columns:
        # If B uses different naming
        for cand in ["category", "Category", "label", "failure_category"]:
            if cand in B.columns:
                cat_col_B = cand
                break
        else:
            print("[WARN] Cannot find category column in overlap files.")
            return
    else:
        cat_col_B = cat_col

    M = A[[key, cat_col]].merge(B[[key, cat_col_B]], on=key, how="inner", suffixes=("_A", "_B"))
    M["cat_A"] = M[f"{cat_col}_A"].apply(_clean_cat)
    M["cat_B"] = M[f"{cat_col_B}_B"].apply(_clean_cat)

    # Drop missing
    M = M.dropna(subset=["cat_A", "cat_B"]).copy()
    if len(M) == 0:
        print("[WARN] No valid category pairs after dropping missing.")
        return

    # Fixed label order (consistent with taxonomy)
    labels = ["Action", "Ambiguous", "Attribute", "Context", "Count", "Object", "Spatial"]
    # Keep labels observed or in the predefined list
    yA = M["cat_A"].tolist()
    yB = M["cat_B"].tolist()

    # Percent agreement
    acc = np.mean([a == b for a, b in zip(yA, yB)]) * 100.0
    kappa = cohen_kappa_score(yA, yB, labels=labels)

    print("==========================================================")
    print("Category agreement / Cohen's kappa (from script)")
    print("==========================================================")
    print(f"N used: {len(M)}")
    print(f"Percent agreement: {acc:.2f}%")
    print(f"Cohen's kappa: {kappa:.4f}")

    # Confusion matrix
    cm = confusion_matrix(yA, yB, labels=labels)

    # ---- Plot confusion heatmap (blue theme + colorbar)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", zorder=2)

    # Colorbar (paper-style)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=COLOR_GRAY)
    cbar.set_label("Count", color=COLOR_GRAY, fontsize=9)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate counts (switch text color by cell intensity)
    maxv = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v != 0:
                txt_color = "white" if v > 0.5 * maxv else "#333333"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=9, color=txt_color, fontweight="bold")

    setup_axis(ax, "Annotator B", "Annotator A")
    ax.set_title("Overlap Confusion Matrix (Category)", fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_overlap_confusion_category_clean.png"), dpi=300)
    plt.savefig(os.path.join(OUTDIR, "fig_overlap_confusion_category_clean.pdf"))
    plt.close()
    print("[Output] Saved fig_overlap_confusion_category_clean")

    # ---- Agreement summary: two panels (correct scales)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))

    # Left: percent agreement (0-100)
    axes[0].bar([0], [acc], width=0.6, color="#4E79A7", edgecolor="white", zorder=3)
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["Percent agreement"])
    setup_axis(axes[0], "", "Percent (%)")
    axes[0].set_ylim(0, 100)
    axes[0].text(0, acc + 2, f"{acc:.2f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#333333")

    # Right: kappa (0-1)
    axes[1].bar([0], [kappa], width=0.6, color="#4E79A7", edgecolor="white", zorder=3)
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["Cohen's kappa"])
    setup_axis(axes[1], "", "Kappa")
    axes[1].set_ylim(0, 1.0)
    axes[1].text(0, kappa + 0.03, f"{kappa:.3f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#333333")

    plt.suptitle("Overlap Agreement Summary", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_overlap_agreement_summary_clean.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "fig_overlap_agreement_summary_clean.pdf"), bbox_inches="tight")
    plt.close()
    print("[Output] Saved fig_overlap_agreement_summary_clean")

plot_overlap_agreement_and_confusion()