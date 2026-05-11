"""
characteristic_analysis.py

Characterizes AD subtypes using clinical covariates:
  - Age, education (continuous)
  - Diagnosis, gender, APOE4 (categorical)

Outputs publication-quality figures and statistical test results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns
from scipy.stats import kruskal, chi2_contingency, mannwhitneyu
from itertools import combinations

matplotlib.use("Agg")

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Georgia", "DejaVu Serif", "Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

SUBTYPE_CMAP  = plt.get_cmap("tab10")
APOE_COLORS   = {0: "#4dac26", 1: "#f1b6da", 2: "#d01c8b"}
GENDER_COLORS = {"Male": "#4393c3", "M": "#4393c3",
                 "Female": "#d6604d", "F": "#d6604d"}
DX_COLORS     = {"CN": "#2166ac", "AD": "#d6604d",
                 "MCI": "#f4a582", "EMCI": "#fdae61", "LMCI": "#d73027"}

STRIP_ALPHA  = 0.35
STRIP_SIZE   = 2.5
BOX_ALPHA    = 0.65

# ============================================================
# Paths  — point at AD-only subtypes by default
# ============================================================
SUBTYPE_PATH = "./results/subtype_analysis/AD_only/AD_subtype_with_bestK.csv"
CHAR_PATH    = "./dataset/characteristic.csv"
SAVE_DIR     = "./results/subtype_analysis/characteristic"
os.makedirs(SAVE_DIR, exist_ok=True)

SUBTYPE_COL  = "AD_subtype"   # column written by the updated kmeans script


# ============================================================
# Load & merge
# ============================================================
sub_df  = pd.read_csv(SUBTYPE_PATH)
char_df = pd.read_csv(CHAR_PATH)

sub_df["PTID"]  = sub_df["PTID"].astype(str)
char_df["PTID"] = char_df["PTID"].astype(str)

char_cols = ["PTID", "DX_bl", "AGE", "PTEDUCAT", "PTGENDER", "APOE4", "MMSE", "CDRSB", "ADAS13"]
char_cols = [c for c in char_cols if c in char_df.columns]
char_df   = char_df[char_cols].copy()

df = pd.merge(sub_df, char_df, on="PTID", how="inner", suffixes=("_sub", ""))

# Coerce types
df[SUBTYPE_COL] = pd.to_numeric(df[SUBTYPE_COL], errors="coerce").astype("Int64")
df["AGE"]       = pd.to_numeric(df["AGE"], errors="coerce")
df["PTEDUCAT"]  = pd.to_numeric(df["PTEDUCAT"], errors="coerce")
df["APOE4"]     = pd.to_numeric(df["APOE4"], errors="coerce").astype("Int64")
df["PTGENDER"]  = df["PTGENDER"].astype(str)
df["DX_bl"]     = df["DX_bl"].astype(str)
for c in ["MMSE", "CDRSB", "ADAS13"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

required_core = [SUBTYPE_COL, "AGE", "PTEDUCAT", "APOE4"]
df = df.dropna(subset=required_core).copy()
df[SUBTYPE_COL] = df[SUBTYPE_COL].astype(int)

subtypes = sorted(df[SUBTYPE_COL].unique())
n_sub    = len(subtypes)
sub_labels = [f"AD-{k}" for k in subtypes]
sub_colors = [SUBTYPE_CMAP(i / max(n_sub - 1, 1)) for i in range(n_sub)]

print(f"Subjects after merge: {len(df)}")
print(f"Subtypes: {subtypes}")


# ============================================================
# Helpers
# ============================================================
def _strip_spines(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

def _significance_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def _add_pairwise_brackets(ax, pairs_pvals, x_positions, y_base, dy=0.05):
    """Draw significance brackets above boxplots for pairwise comparisons."""
    y = y_base
    for (i, j), p in pairs_pvals:
        xi, xj = x_positions[i], x_positions[j]
        stars = _significance_stars(p)
        ax.plot([xi, xi, xj, xj], [y, y + dy*0.4, y + dy*0.4, y],
                lw=0.9, color="#444444")
        ax.text((xi + xj) / 2, y + dy*0.45, stars,
                ha="center", va="bottom", fontsize=9, color="#444444")
        y += dy * 0.9


# ============================================================
# Statistical tests
# ============================================================
stats_rows = []

# --- Continuous: Kruskal-Wallis + pairwise Mann-Whitney ---
cont_vars = [v for v in ["AGE", "PTEDUCAT", "MMSE", "CDRSB", "ADAS13"]
             if v in df.columns]

pairwise_results = {}   # var -> list of ((i,j), p)

for var in cont_vars:
    groups = [df.loc[df[SUBTYPE_COL] == k, var].dropna().values for k in subtypes]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        continue
    stat, p = kruskal(*groups)
    stats_rows.append({"variable": var, "test": "Kruskal-Wallis",
                        "statistic": stat, "p_value": p})

    # Pairwise MWU for bracket annotation
    pw = []
    for (a, b) in combinations(range(len(subtypes)), 2):
        ga = df.loc[df[SUBTYPE_COL] == subtypes[a], var].dropna()
        gb = df.loc[df[SUBTYPE_COL] == subtypes[b], var].dropna()
        if len(ga) > 0 and len(gb) > 0:
            _, pp = mannwhitneyu(ga, gb, alternative="two-sided")
            pw.append(((a, b), pp))
    pairwise_results[var] = pw

# --- Categorical: Chi-squared ---
cat_vars = [v for v in ["DX_bl", "PTGENDER", "APOE4"] if v in df.columns]
for var in cat_vars:
    tbl = pd.crosstab(df[SUBTYPE_COL], df[var])
    if tbl.shape[0] >= 2 and tbl.shape[1] >= 2:
        stat, p, dof, _ = chi2_contingency(tbl)
        stats_rows.append({"variable": var, "test": "Chi-squared",
                            "statistic": stat, "p_value": p, "dof": dof})

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(SAVE_DIR, "subtype_stats.csv"), index=False)
print("\n=== Statistical tests ===")
print(stats_df.to_string(index=False))


# ============================================================
# Summary tables
# ============================================================
cont_summary = df.groupby(SUBTYPE_COL)[cont_vars].agg(
    ["mean", "std", "median", "count"]).round(2)
cont_summary.to_csv(os.path.join(SAVE_DIR, "continuous_summary.csv"))

for var in cat_vars:
    cnt = pd.crosstab(df[SUBTYPE_COL], df[var])
    pct = (cnt.div(cnt.sum(axis=1), axis=0) * 100).round(1)
    cnt.to_csv(os.path.join(SAVE_DIR, f"{var}_count.csv"))
    pct.to_csv(os.path.join(SAVE_DIR, f"{var}_percent.csv"))


# ============================================================
# FIGURE 1 — Continuous variables: box + strip grid
# ============================================================
n_cont = len(cont_vars)
fig, axes = plt.subplots(1, n_cont, figsize=(3.8 * n_cont, 4.8), sharey=False)
if n_cont == 1:
    axes = [axes]

for ax, var in zip(axes, cont_vars):
    data_by_sub = [df.loc[df[SUBTYPE_COL] == k, var].dropna().values
                   for k in subtypes]

    bp = ax.boxplot(
        data_by_sub, patch_artist=True, notch=False, widths=0.55,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=0.9, color="#555555"),
        capprops=dict(linewidth=0.9, color="#555555"),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.3,
                        markerfacecolor="#888888", markeredgecolor="none"),
    )
    for patch, col in zip(bp["boxes"], sub_colors):
        patch.set_facecolor(col)
        patch.set_alpha(BOX_ALPHA)
        patch.set_linewidth(0.7)

    # Strip jitter
    for i, (vals, col) in enumerate(zip(data_by_sub, sub_colors)):
        jitter = np.random.default_rng(42 + i).uniform(-0.18, 0.18, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   color=col, alpha=STRIP_ALPHA, s=STRIP_SIZE ** 2,
                   edgecolors="none", zorder=3)

    # Pairwise brackets
    if var in pairwise_results and pairwise_results[var]:
        all_vals = np.concatenate([v for v in data_by_sub if len(v)])
        y_top = np.nanpercentile(all_vals, 97)
        dy    = (np.nanmax(all_vals) - np.nanmin(all_vals)) * 0.08
        _add_pairwise_brackets(
            ax, pairwise_results[var],
            x_positions={i: i + 1 for i in range(n_sub)},
            y_base=y_top + dy * 0.3, dy=dy,
        )

    # Global KW p
    kw_row = stats_df[(stats_df["variable"] == var) &
                       (stats_df["test"] == "Kruskal-Wallis")]
    if not kw_row.empty:
        p_kw = kw_row["p_value"].values[0]
        ax.set_title(f"{var}\n(KW p={p_kw:.3f})", pad=8)
    else:
        ax.set_title(var, pad=8)

    ax.set_xticks(range(1, n_sub + 1))
    ax.set_xticklabels(sub_labels, fontsize=9)
    ax.set_ylabel(var)
    _strip_spines(ax)

fig.suptitle("Continuous clinical variables by AD subtype",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout(pad=2.0)
fig.savefig(os.path.join(SAVE_DIR, "continuous_boxplots.png"), dpi=150)
plt.close(fig)
print("Saved: continuous_boxplots.png")


# ============================================================
# FIGURE 2 — Categorical: stacked + grouped bars  (2 rows)
# ============================================================
cat_meta = {
    "APOE4":    ("APOE4 allele count", APOE_COLORS,   "APOE4"),
    "PTGENDER": ("Sex",                GENDER_COLORS,  "Gender"),
    "DX_bl":    ("Baseline diagnosis", DX_COLORS,      "Diagnosis"),
}
cat_meta = {k: v for k, v in cat_meta.items() if k in df.columns}
n_cat = len(cat_meta)

fig, axes = plt.subplots(2, n_cat, figsize=(5 * n_cat, 8.5))
if n_cat == 1:
    axes = axes.reshape(2, 1)

x = np.arange(n_sub)

for col_i, (var, (label, color_map, leg_title)) in enumerate(cat_meta.items()):
    cnt = pd.crosstab(df[SUBTYPE_COL], df[var])
    pct = cnt.div(cnt.sum(axis=1), axis=0) * 100
    categories = cnt.columns.tolist()
    colors = [color_map.get(str(c), f"C{i}") for i, c in enumerate(categories)]

    # Row 0: stacked bar (proportion)
    ax = axes[0, col_i]
    bottom = np.zeros(n_sub)
    for cat, col in zip(categories, colors):
        vals = pct[cat].values
        bars = ax.bar(x, vals, bottom=bottom, color=col,
                      label=str(cat), edgecolor="white", linewidth=0.5, alpha=0.85)
        # Label slices ≥ 10%
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 10:
                ax.text(xi, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if v > 20 else "#333333", fontweight="bold")
        bottom += vals

    # Chi2 p-value in title
    chi_row = stats_df[(stats_df["variable"] == var) &
                        (stats_df["test"] == "Chi-squared")]
    p_str = f"χ² p={chi_row['p_value'].values[0]:.3f}" if not chi_row.empty else ""
    ax.set_title(f"{label}\n({p_str})", pad=8)
    ax.set_xticks(x); ax.set_xticklabels(sub_labels, fontsize=9)
    ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 112)
    ax.legend(title=leg_title, bbox_to_anchor=(1.01, 1), loc="upper left",
              title_fontsize=8, fontsize=8)
    _strip_spines(ax)

    # Row 1: grouped bar (count)
    ax = axes[1, col_i]
    n_cat_vals = len(categories)
    width = 0.7 / n_cat_vals
    for ci, (cat, col) in enumerate(zip(categories, colors)):
        offset = (ci - (n_cat_vals - 1) / 2) * width
        ax.bar(x + offset, cnt[cat].values, width=width, color=col,
               label=str(cat), edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_title(f"{label} — counts", pad=8)
    ax.set_xticks(x); ax.set_xticklabels(sub_labels, fontsize=9)
    ax.set_ylabel("Count")
    ax.legend(title=leg_title, bbox_to_anchor=(1.01, 1), loc="upper left",
              title_fontsize=8, fontsize=8)
    _strip_spines(ax)

fig.suptitle("Categorical clinical variables by AD subtype",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout(pad=2.5)
fig.savefig(os.path.join(SAVE_DIR, "categorical_bars.png"), dpi=150)
plt.close(fig)
print("Saved: categorical_bars.png")


# ============================================================
# FIGURE 3 — Summary table heatmap
# ============================================================
# Build a compact numeric summary matrix: rows = subtypes, cols = cont_vars
heatmap_data = pd.DataFrame(index=sub_labels)
for var in cont_vars:
    col_vals = []
    for k in subtypes:
        v = df.loc[df[SUBTYPE_COL] == k, var].mean()
        col_vals.append(v)
    # Z-score across subtypes for visual comparability
    arr = np.array(col_vals, dtype=float)
    if arr.std() > 0:
        arr = (arr - arr.mean()) / arr.std()
    heatmap_data[var] = arr

fig, ax = plt.subplots(figsize=(max(5, len(cont_vars) * 1.4), 2.5 + 0.5 * n_sub))
im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdBu_r",
               vmin=-2, vmax=2)
ax.set_xticks(range(len(cont_vars)))
ax.set_xticklabels(cont_vars, fontsize=10, rotation=20, ha="right")
ax.set_yticks(range(n_sub))
ax.set_yticklabels(sub_labels, fontsize=10)

# Annotate cells
for i in range(n_sub):
    for j, var in enumerate(cont_vars):
        raw_val = df.loc[df[SUBTYPE_COL] == subtypes[i], var].mean()
        ax.text(j, i, f"{raw_val:.1f}", ha="center", va="center",
                fontsize=8.5, color="black")

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
cbar.set_label("Z-score across subtypes", fontsize=9)
ax.set_title("Clinical summary heatmap (mean; colour = z-score across subtypes)",
             pad=10)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "clinical_summary_heatmap.png"), dpi=150)
plt.close(fig)
print("Saved: clinical_summary_heatmap.png")


# ============================================================
# Save merged dataframe
# ============================================================
df.to_csv(os.path.join(SAVE_DIR, "subtype_characteristics_merged.csv"), index=False)
print(f"\nAll outputs saved to: {SAVE_DIR}")

# import os

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from scipy.stats import kruskal, chi2_contingency


# # subtype_path = "./results/subtype_analysis/subtype_with_bestK.csv"
# subtype_path ="./results/subtype_analysis/AD_only/AD_only_subtype_with_bestK.csv"
# char_path = "./dataset/characteristic.csv"
# save_dir = "./results/subtype_analysis"
# os.makedirs(save_dir, exist_ok=True)


# # ------------------------------------------------------------
# # Load and merge
# # ------------------------------------------------------------
# sub_df = pd.read_csv(subtype_path)
# char_df = pd.read_csv(char_path)

# sub_df["PTID"] = sub_df["PTID"].astype(str)
# char_df["PTID"] = char_df["PTID"].astype(str)

# # Keep only characteristic columns needed for analysis.
# char_cols = [
#     "PTID",
#     "DX_bl",
#     "AGE",
#     "PTEDUCAT",
#     "PTGENDER",
#     "APOE4",
# ]

# char_cols = [c for c in char_cols if c in char_df.columns]
# char_df = char_df[char_cols].copy()

# df = pd.merge(sub_df, char_df, on="PTID", how="inner", suffixes=("_sub", ""))

# # Required fields
# required_cols = [
#     "subtype",
#     "DX_bl",
#     "AGE",
#     "PTEDUCAT",
#     "PTGENDER",
#     "APOE4",
# ]

# df = df.dropna(subset=required_cols).copy()

# df["subtype"] = df["subtype"].astype(int)
# df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
# df["PTEDUCAT"] = pd.to_numeric(df["PTEDUCAT"], errors="coerce")
# df["PTGENDER"] = df["PTGENDER"].astype(str)
# df["APOE4"] = pd.to_numeric(df["APOE4"], errors="coerce").astype(int).astype(str)
# df["DX_bl"] = df["DX_bl"].astype(str)

# df = df.dropna(subset=["AGE", "PTEDUCAT", "APOE4"]).copy()

# print("Subjects after merge/dropna:", len(df))
# print("Subtypes:", sorted(df["subtype"].unique()))


# # ------------------------------------------------------------
# # Continuous summaries
# # ------------------------------------------------------------
# cont_table = df.groupby("subtype")[["AGE", "PTEDUCAT"]].agg(["mean", "std", "median", "count"])
# cont_table.to_csv(os.path.join(save_dir, "continuous_summary.csv"))

# print("\n=== Continuous Summary ===")
# print(cont_table)


# # ------------------------------------------------------------
# # Categorical summaries
# # ------------------------------------------------------------
# def save_crosstab(var, prefix):
#     cnt = pd.crosstab(df["subtype"], df[var])
#     pct = pd.crosstab(df["subtype"], df[var], normalize="index") * 100

#     cnt.to_csv(os.path.join(save_dir, f"{prefix}_count.csv"))
#     pct.to_csv(os.path.join(save_dir, f"{prefix}_percent.csv"))

#     print(f"\n=== {prefix} count ===")
#     print(cnt)

#     print(f"\n=== {prefix} percent ===")
#     print(pct)

#     return cnt, pct


# dx_cnt, dx_pct = save_crosstab("DX_bl", "diagnosis")
# gender_cnt, gender_pct = save_crosstab("PTGENDER", "gender")
# apoe_cnt, apoe_pct = save_crosstab("APOE4", "apoe")


# # ------------------------------------------------------------
# # Statistical tests
# # ------------------------------------------------------------
# stats_rows = []

# # Kruskal-Wallis for continuous variables
# for var in ["AGE", "PTEDUCAT"]:
#     groups = [
#         g[var].dropna().values
#         for _, g in df.groupby("subtype")
#         if len(g[var].dropna()) > 0
#     ]

#     if len(groups) >= 2:
#         stat, p = kruskal(*groups)
#         stats_rows.append({
#             "variable": var,
#             "test": "Kruskal-Wallis",
#             "statistic": stat,
#             "p_value": p,
#         })

# # Chi-square for categorical variables
# for var in ["DX_bl", "PTGENDER", "APOE4"]:
#     table = pd.crosstab(df["subtype"], df[var])

#     if table.shape[0] >= 2 and table.shape[1] >= 2:
#         stat, p, dof, expected = chi2_contingency(table)
#         stats_rows.append({
#             "variable": var,
#             "test": "Chi-square",
#             "statistic": stat,
#             "p_value": p,
#             "dof": dof,
#         })

# stats_df = pd.DataFrame(stats_rows)
# stats_df.to_csv(os.path.join(save_dir, "subtype_characteristic_tests.csv"), index=False)

# print("\n=== Statistical Tests ===")
# print(stats_df)


# # ------------------------------------------------------------
# # Plots
# # ------------------------------------------------------------
# sns.set(style="whitegrid")


# # AGE
# plt.figure(figsize=(6, 4))
# sns.boxplot(data=df, x="subtype", y="AGE")
# sns.stripplot(data=df, x="subtype", y="AGE", color="black", alpha=0.25, size=2)
# plt.title("Age by subtype")
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "age_boxplot.png"), dpi=150)
# plt.close()


# # EDUCATION
# plt.figure(figsize=(6, 4))
# sns.boxplot(data=df, x="subtype", y="PTEDUCAT")
# sns.stripplot(data=df, x="subtype", y="PTEDUCAT", color="black", alpha=0.25, size=2)
# plt.title("Education by subtype")
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "education_boxplot.png"), dpi=150)
# plt.close()


# def plot_percent_bar(pct_table, var_name, title, filename):
#     plot_df = pct_table.reset_index().melt(
#         id_vars="subtype",
#         var_name=var_name,
#         value_name="Percent",
#     )

#     plt.figure(figsize=(7, 4))
#     sns.barplot(data=plot_df, x="subtype", y="Percent", hue=var_name)
#     plt.title(title)
#     plt.ylabel("Percent within subtype")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, filename), dpi=150)
#     plt.close()


# plot_percent_bar(
#     dx_pct,
#     var_name="Diagnosis",
#     title="Diagnosis distribution by subtype (%)",
#     filename="diagnosis_barplot.png",
# )

# plot_percent_bar(
#     gender_pct,
#     var_name="Gender",
#     title="Gender distribution by subtype (%)",
#     filename="gender_barplot.png",
# )

# plot_percent_bar(
#     apoe_pct,
#     var_name="APOE4",
#     title="APOE4 distribution by subtype (%)",
#     filename="apoe_barplot.png",
# )


# # ------------------------------------------------------------
# # Save merged analysis dataframe
# # ------------------------------------------------------------
# df.to_csv(os.path.join(save_dir, "subtype_characteristics_merged.csv"), index=False)

# print("\nDone. Results saved to:", save_dir)