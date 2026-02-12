import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置与配色 (Configuration)
# ==========================================
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR if (_SCRIPT_DIR / "data").exists() else _SCRIPT_DIR.parent

# 输入文件
SUMMARY_CSV = str(PROJECT_ROOT / "outputs" / "summary" / "summary_subset_results.csv")
ASSIGN_CSV  = str(PROJECT_ROOT / "failure_analysis" / "analysis_all" / "annotations_clean.csv")

# Bootstrap data files
HITS_FILES = {
    "max": str(PROJECT_ROOT / "outputs" / "subset_hits" / "subset_hits_Action_n24_max_seed42.csv"),
    "mean": str(PROJECT_ROOT / "outputs" / "subset_hits" / "subset_hits_Action_n24_mean_seed42.csv"),
    "logsumexp": str(PROJECT_ROOT / "outputs" / "subset_hits" / "subset_hits_Action_n24_logsumexp_seed42.csv"),
}

OUTDIR = str(PROJECT_ROOT / "outputs" / "figures")
os.makedirs(OUTDIR, exist_ok=True)

# --- 美化核心：自定义配色方案 (Tableau 风格) ---
# 这种配色比默认的 RGB 更加沉稳、专业
COLORS = {
    "max":       "#4E79A7",  # 深蓝 (Deep Blue)
    "mean":      "#F28E2B",  # 柔和橙 (Muted Orange)
    "logsumexp": "#59A14F",  # 森绿 (Forest Green)
}
COLOR_GRAY = "#595959"       # 字体/边框灰色

# --- 全局绘图风格设置 ---
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans'] # 优先使用 Arial
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = COLOR_GRAY
plt.rcParams['xtick.color'] = COLOR_GRAY
plt.rcParams['ytick.color'] = COLOR_GRAY
plt.rcParams['text.color'] = 'black'
plt.rcParams['figure.dpi'] = 150 

# ==========================================
# 2. 数据加载与处理
# ==========================================
def generate_dummy_data():
    """如果没有 CSV，生成模拟数据以供测试"""
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

# 尝试读取 CSV，失败则使用模拟数据
if os.path.exists(SUMMARY_CSV):
    df = pd.read_csv(SUMMARY_CSV)
else:
    df = generate_dummy_data()

# 标准化类别名称，确保顺序一致
subset_order = ["Object+Attribute", "Object", "Attribute", "Action"]
pool_order   = ["max", "mean", "logsumexp"]

# 处理一下 DataFrame 中的类别名，防止 CSV 里写的是 "Object,Attribute" 而这里用 "+"
# 这是一个常见的数据清洗步骤
df["categories"] = df["categories"].str.replace(",", "+")

# ==========================================
# 3. 绘图函数 (核心美化逻辑)
# ==========================================

def setup_axis(ax, xlabel, ylabel):
    """通用的坐标轴美化"""
    # 隐藏上边和右边的边框 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加淡灰色网格，zorder=0 保证网格在柱子后面
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)
    
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=8, color='#333333')
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=8, color='#333333')
def grouped_bar_beautiful(metric, fname, ylabel):
    """绘制分组柱状图，带数值标注（仅修改图例位置）"""
    # 准备数据
    cats = [c for c in subset_order if c in df["categories"].unique()]
    x = np.arange(len(cats))
    width = 0.25  # 柱子宽度
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 循环绘制每一种 pooling
    for i, p in enumerate(pool_order):
        vals = []
        for cat in cats:
            row = df[(df["categories"] == cat) & (df["pooling"] == p)]
            if len(row) == 0:
                vals.append(0.0)
            else:
                vals.append(float(row.iloc[0][metric]))
        
        # 计算 x 轴位置
        x_pos = x + (i - 1) * width
        
        # 绘制柱子（配色、样式完全不变）
        bars = ax.bar(
            x_pos, vals, width, 
            label=p, 
            color=COLORS[p], 
            edgecolor='white',  
            linewidth=0.7,
            zorder=3            
        )
        
        # 数值标签（完全不变）
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

    # 坐标轴设置（完全不变）
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=0)
    setup_axis(ax, "Failure Subset", ylabel)
    
    # --- 仅修改这部分：动态判断图例位置 ---
    # 计算左右子集的数值总和
    left_sum = df[df["categories"].isin(subset_order[:2])][metric].sum()  # 左：Object+Attribute/Object
    right_sum = df[df["categories"].isin(subset_order[2:])][metric].sum() # 右：Attribute/Action
    # 左边高→upper right，右边高→upper left
    legend_loc = "upper right" if left_sum > right_sum else "upper left"
    
    ax.legend(title="Pooling Strategy", frameon=False, loc=legend_loc)

    # 保存（完全不变）
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname + ".png"), dpi=300)
    plt.savefig(os.path.join(OUTDIR, fname + ".pdf"))
    plt.close()
    print(f"[Output] Saved {fname} (legend at {legend_loc})")

# ==========================================
# 4. 执行绘图
# ==========================================

# --- 图 1 & 2: 分组柱状图 (R@5 和 R@10) ---
grouped_bar_beautiful("delta_pp_R@5",  "fig_pooling_ablation_delta_r5_clean",  "Recall@5 Gain (pp)")
grouped_bar_beautiful("delta_pp_R@10", "fig_pooling_ablation_delta_r10_clean", "Recall@10 Gain (pp)")

# --- 图 3: Best Pooling per Subset ---
# 筛选最佳策略
best = (
    df.sort_values(by="delta_pp_R@10", ascending=False)
      .drop_duplicates(subset=["categories"], keep="first")
)
# 强制排序
best = best.set_index("categories").reindex(subset_order).reset_index()

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(best))
# 根据最佳 pooling 策略来分配颜色
bar_colors = [COLORS[p] for p in best["pooling"]]

bars = ax.bar(
    x, best["delta_pp_R@10"], 
    width=0.6, 
    color=bar_colors,
    edgecolor='white',
    zorder=3
)

# 标注数值
for bar, val, pooling in zip(bars, best["delta_pp_R@10"], best["pooling"]):
    height = bar.get_height()
    y_offset = 0.2 if height < 0.5 else height + 0.2
    
    # 标数值
    ax.text(
        bar.get_x() + bar.get_width()/2, y_offset, 
        f"+{val:.1f}", 
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )
    # 在柱子内部标 pooling 名字 (如果空间够)
    if height > 2:
        ax.text(
            bar.get_x() + bar.get_width()/2, height/2, 
            pooling, 
            ha='center', va='center', color='white', fontsize=9, fontweight='bold'
        )
    else:
        # 空间不够写在上面
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

# --- 图 4: Bootstrap (如果文件存在) ---
# 这里我加了一个模拟逻辑，确保你能看到图的样子
def plot_bootstrap_beautiful():
    B = 2000
    rng = np.random.default_rng(42)
    delta_samples = {}

    # 真实读取 + bootstrap ΔR@10
    for pooling, path in HITS_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing hits file: {path}")
            continue

        hits = pd.read_csv(path)

        # 你的列名是 baseline_hit@10 / improved_hit@10
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

    # 如果还是空，直接退出（避免 KeyError）
    if len(delta_samples) == 0:
        print("[WARN] No bootstrap samples computed. Check HITS_FILES paths.")
        return

    # 只画存在的 pooling（保持顺序）
    keys = [p for p in pool_order if p in delta_samples]
    data_to_plot = [delta_samples[p] for p in keys]

    fig, ax = plt.subplots(figsize=(6, 5))

    # 箱线图美化
    boxprops = dict(linewidth=1.5, color=COLOR_GRAY)
    medianprops = dict(linewidth=2, color='#D32F2F')
    whiskerprops = dict(linewidth=1.5, color=COLOR_GRAY)
    capprops = dict(linewidth=1.5, color=COLOR_GRAY)

    bp = ax.boxplot(
        data_to_plot,
        tick_labels=keys,        # Matplotlib 3.9+ 用 tick_labels
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

    # 可选：画一条 0 参考线
    ax.axhline(0, linestyle="--", linewidth=1, color=COLOR_GRAY, alpha=0.6, zorder=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_bootstrap_action_delta_r10_clean.png"), dpi=300)
    plt.savefig(os.path.join(OUTDIR, "fig_bootstrap_action_delta_r10_clean.pdf"))
    plt.close()
    print(f"[Output] Saved fig_bootstrap_action_delta_r10_clean")

plot_bootstrap_beautiful()