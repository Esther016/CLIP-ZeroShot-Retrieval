import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- 配置 Times New Roman 字体 ----------
# 尝试使用系统自带的 Times New Roman，如果失败，matplotlib 会自动回退到 serif
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "stix"  # 使数学符号也接近 Times 风格

# ---------- 加载数据 (保持你的原始逻辑) ----------
df = pd.read_csv("assign_all.csv")
counts = df["category"].value_counts().sort_values(ascending=False)


# ---------- 绘图设置 ----------
fig, ax = plt.subplots(figsize=(7, 4.5))

# 使用更具学术感的深蓝色，并添加轻微透明度
color = "#3c77b1" 
counts.plot(kind="bar", color=color, alpha=0.85, width=0.7, ax=ax)

# ---------- 细节打磨 ----------
# 1. 修改 y 轴标签 (包含总样本数 n)
total_n = counts.sum()
ax.set_ylabel(f"Number of annotated failures (n = {total_n})", fontsize=12)
ax.set_xlabel("Failure Category", fontsize=12)

# 2. 去掉标题 (如你所愿，通过 Caption 传达)
# plt.title("...") 

# 3. 移除上方和右方的边框 (Spines)，让图表更清爽
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 4. 优化刻度线：将类别标签旋转为 45 度以便阅读，并调整字体大小
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=10)

# 5. 在每个柱子上方添加数值标注 (Data Labels)
for i, v in enumerate(counts):
    ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

# 6. 添加水平网格线，增强可读性 (仅 y 轴)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True) # 确保网格线在柱子后面

# ---------- 导出 ----------
plt.tight_layout()
# 建议导出为 PDF 格式，这样在 LaTeX 中缩放不会失真，且保留矢量字体
plt.savefig("fig_taxonomy_distribution_counts.pdf", bbox_inches='tight')
plt.show()