import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, Normalize

from scipy.stats import kruskal, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from itertools import combinations


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

SUBTYPE_CMAP = plt.get_cmap("tab10")


def _strip_spines(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def _sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _clean_roi(name):
    name = name.replace("_SUVR", "").replace("_VOLUME", "")
    name = name.replace("_suvr", "").replace("_volume", "")
    name = name.replace("_", " ").strip()
    return name


# ============================================================
# Paths — edit these to switch between MRI / PET
# ============================================================
ROI_PATH = "./dataset/PET_whole.csv"  # or "./dataset/MRI_whole.csv"
SUBTYPE_PATH = "./results/subtype_analysis/AD_only/AD_subtype_with_bestK.csv"
OUTPUT_DIR = "./results/subtype_analysis/ROI_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PTID_COL = "PTID"
SUBTYPE_COL = "AD_subtype"

# ROI_KEYWORD = "suvr"       # "suvr" for PET, "volume" for MRI
TOP_K = 15
FDR_ALPHA = 0.05
# MODALITY = "PET"           # label for titles
MODALITY = "MRI"  
if MODALITY == "MRI":
    ROI_KEYWORD = "volume"
else:
    ROI_KEYWORD = "suvr" 
ROI_PATH = f"./dataset/{MODALITY}_whole.csv"  # or "./dataset/MRI_whole.csv" or "./dataset/PET_whole.csv"
# ============================================================
# Load & merge
# ============================================================
roi_df = pd.read_csv(ROI_PATH)
sub_df = pd.read_csv(SUBTYPE_PATH)

roi_df[PTID_COL] = roi_df[PTID_COL].astype(str).str.strip()
sub_df[PTID_COL] = sub_df[PTID_COL].astype(str).str.strip()

df = roi_df.merge(
    sub_df[[PTID_COL, SUBTYPE_COL]],
    on=PTID_COL,
    how="inner",
)

df[SUBTYPE_COL] = pd.to_numeric(df[SUBTYPE_COL], errors="coerce").astype("Int64")
df = df.dropna(subset=[SUBTYPE_COL]).copy()
df[SUBTYPE_COL] = df[SUBTYPE_COL].astype(int)

roi_cols = [c for c in df.columns if ROI_KEYWORD in c.lower()]

print(f"ROI file: {ROI_PATH}")
print(f"Subtype file: {SUBTYPE_PATH}")
print(f"ROI keyword: {ROI_KEYWORD}")
print(f"ROI features found: {len(roi_cols)}")
print(f"Subjects after merge: {len(df)}")

if len(roi_cols) == 0:
    raise ValueError(
        f"No ROI columns found using ROI_KEYWORD='{ROI_KEYWORD}'. "
        f"Check ROI_PATH and ROI_KEYWORD."
    )

# Convert ROI columns to numeric
for c in roi_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=roi_cols + [SUBTYPE_COL]).copy()
print(f"After dropna on ROI columns: {len(df)}")

subtypes = sorted(df[SUBTYPE_COL].unique())
n_sub = len(subtypes)

if n_sub < 2:
    raise ValueError(f"Need at least 2 AD subtypes. Found: {subtypes}")

sub_labels = [f"AD-{k}" for k in subtypes]
sub_colors = [SUBTYPE_CMAP(i / max(n_sub - 1, 1)) for i in range(n_sub)]

print(f"Subtypes: {subtypes}")
print("Subtype counts:")
print(df[SUBTYPE_COL].value_counts().sort_index().to_string())


# ============================================================
# Remove raw constant ROI columns before scaling
# ============================================================
raw_constant_rois = [
    c for c in roi_cols
    if df[c].nunique(dropna=True) <= 1
]

if len(raw_constant_rois) > 0:
    print(f"\nRaw constant ROI columns skipped: {len(raw_constant_rois)}")
    print("Examples:", raw_constant_rois[:10])

roi_cols = [c for c in roi_cols if c not in raw_constant_rois]

if len(roi_cols) == 0:
    raise ValueError(
        "All ROI columns are constant after merging with AD subtype file. "
        "No ROI analysis can be performed."
    )

print(f"ROI features retained after removing raw constants: {len(roi_cols)}")


# ============================================================
# Standardise ROI values
# ============================================================
scaler = StandardScaler()
df_z = df.copy()
df_z[roi_cols] = scaler.fit_transform(df[roi_cols])

# Remove any columns that became non-finite after scaling
finite_roi_cols = []
bad_scaled_rois = []

for c in roi_cols:
    vals = df_z[c].values
    if np.all(np.isfinite(vals)):
        finite_roi_cols.append(c)
    else:
        bad_scaled_rois.append(c)

if bad_scaled_rois:
    print(f"\nNon-finite scaled ROI columns skipped: {len(bad_scaled_rois)}")
    print("Examples:", bad_scaled_rois[:10])

roi_cols = finite_roi_cols

if len(roi_cols) == 0:
    raise ValueError("No finite ROI columns remain after z-scoring.")


# ============================================================
# Kruskal-Wallis + FDR
# ============================================================
kw_results = {}
skipped_empty = []
skipped_constant = []

for roi in roi_cols:
    groups = [
        df_z.loc[df_z[SUBTYPE_COL] == k, roi].dropna().values
        for k in subtypes
    ]

    if any(len(g) == 0 for g in groups):
        skipped_empty.append(roi)
        continue

    pooled = np.concatenate(groups)

    # Kruskal-Wallis fails if all values are identical.
    if len(np.unique(pooled)) <= 1:
        skipped_constant.append(roi)
        continue

    try:
        stat, p = kruskal(*groups)
    except ValueError as e:
        if "All numbers are identical" in str(e):
            skipped_constant.append(roi)
            continue
        raise e

    kw_results[roi] = {
        "kw_stat": stat,
        "p_value": p,
    }

print(f"\nSkipped empty-group ROIs: {len(skipped_empty)}")
if skipped_empty:
    print("Examples:", skipped_empty[:10])

print(f"Skipped constant ROIs at KW stage: {len(skipped_constant)}")
if skipped_constant:
    print("Examples:", skipped_constant[:10])

kw_df = pd.DataFrame(kw_results).T.reset_index().rename(columns={"index": "ROI"})

if kw_df.empty:
    raise ValueError(
        "No valid ROI columns remained after Kruskal-Wallis filtering. "
        "This usually means all ROI columns were constant or invalid after merging."
    )

kw_df = kw_df.sort_values("p_value").reset_index(drop=True)

reject, p_corr, _, _ = multipletests(
    kw_df["p_value"].values,
    alpha=FDR_ALPHA,
    method="fdr_bh",
)

kw_df["p_corrected"] = p_corr
kw_df["significant"] = reject
kw_df["-log10p"] = -np.log10(kw_df["p_value"].clip(1e-300))
kw_df["-log10p_fdr"] = -np.log10(kw_df["p_corrected"].clip(1e-300))

kw_df.to_csv(
    os.path.join(OUTPUT_DIR, f"{MODALITY}_kruskal_results.csv"),
    index=False,
)

top_rois = kw_df.loc[kw_df["significant"], "ROI"].values[:TOP_K].tolist()

if len(top_rois) == 0:
    top_rois = kw_df["ROI"].values[:TOP_K].tolist()

print(f"\nSignificant ROIs (FDR<{FDR_ALPHA}): {kw_df['significant'].sum()}")
print("\nTop Kruskal-Wallis ROIs:")
print(kw_df.head(10).to_string(index=False))


# ============================================================
# Cohen's d + Mann-Whitney U for K=2 only
# ============================================================
mwu_df = None

if n_sub == 2:
    g0 = df_z[df_z[SUBTYPE_COL] == subtypes[0]]
    g1 = df_z[df_z[SUBTYPE_COL] == subtypes[1]]

    mwu_rows = {}
    skipped_mwu_constant = []

    for roi in roi_cols:
        x0 = g0[roi].dropna().values
        x1 = g1[roi].dropna().values

        if len(x0) == 0 or len(x1) == 0:
            continue

        pooled = np.concatenate([x0, x1])

        if len(np.unique(pooled)) <= 1:
            skipped_mwu_constant.append(roi)
            continue

        try:
            stat, p = mannwhitneyu(x0, x1, alternative="two-sided")
        except ValueError:
            skipped_mwu_constant.append(roi)
            continue

        pooled_sd = np.sqrt((np.std(x0, ddof=1) ** 2 + np.std(x1, ddof=1) ** 2) / 2)

        if pooled_sd == 0 or not np.isfinite(pooled_sd):
            d = 0.0
        else:
            d = (np.mean(x0) - np.mean(x1)) / pooled_sd

        mwu_rows[roi] = {
            "mwu_stat": stat,
            "p_mwu": p,
            "cohen_d": d,
        }

    print(f"\nSkipped constant ROIs at MWU stage: {len(skipped_mwu_constant)}")
    if skipped_mwu_constant:
        print("Examples:", skipped_mwu_constant[:10])

    mwu_df = pd.DataFrame(mwu_rows).T.reset_index().rename(columns={"index": "ROI"})

    if not mwu_df.empty:
        reject_m, p_corr_m, _, _ = multipletests(
            mwu_df["p_mwu"].values,
            alpha=FDR_ALPHA,
            method="fdr_bh",
        )

        mwu_df["p_mwu_fdr"] = p_corr_m
        mwu_df["sig_mwu"] = reject_m
        mwu_df["-log10p_mwu"] = -np.log10(mwu_df["p_mwu"].clip(1e-300))
        mwu_df = mwu_df.sort_values("p_mwu").reset_index(drop=True)

        mwu_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{MODALITY}_mwu_effect.csv"),
            index=False,
        )
    else:
        print("Warning: MWU results are empty.")


# ============================================================
# Mean ROI per subtype
# ============================================================
roi_mean = df_z.groupby(SUBTYPE_COL)[roi_cols].mean()
roi_mean.to_csv(os.path.join(OUTPUT_DIR, f"{MODALITY}_roi_mean_by_subtype.csv"))


# ============================================================
# FIGURE 1 — Heatmap: top significant ROIs × subtype
# ============================================================
n_heat = min(
    40,
    len(top_rois) if kw_df["significant"].sum() > 0 else len(roi_cols),
)

heat_rois = kw_df["ROI"].values[:n_heat]
heat_data = roi_mean[heat_rois]  # shape [n_sub, n_heat]
clean_names = [_clean_roi(r) for r in heat_rois]

fig_h = max(4.5, 0.35 * n_heat + 1.5)
fig, ax = plt.subplots(figsize=(max(6, n_sub * 1.8), fig_h))

vmax = np.abs(heat_data.values).max()

if vmax == 0 or not np.isfinite(vmax):
    vmax = 1.0

norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(heat_data.values.T, aspect="auto", cmap="RdBu_r", norm=norm)

ax.set_xticks(range(n_sub))
ax.set_xticklabels(sub_labels, fontsize=10)
ax.set_yticks(range(n_heat))
ax.set_yticklabels(clean_names, fontsize=7.5)
ax.set_xlabel("AD subtype")
ax.set_title(
    f"{MODALITY} ROI mean (z-scored) — top {n_heat} by KW p-value",
    pad=10,
)

# Mark significant ROIs
for ri, roi in enumerate(heat_rois):
    row = kw_df[kw_df["ROI"] == roi]
    if not row.empty and bool(row["significant"].values[0]):
        ax.text(
            n_sub - 0.42,
            ri,
            "✱",
            ha="left",
            va="center",
            fontsize=8,
            color="#c0392b",
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.06)
cbar.set_label("Mean z-score", fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"{MODALITY}_heatmap.png"), dpi=150)
plt.close(fig)
print("Saved: heatmap")


# ============================================================
# FIGURE 2 — Volcano plot
# KW: -log10 p vs mean absolute subtype difference
# ============================================================
grand_mean = df_z[roi_cols].mean()
max_abs_diff = (roi_mean - grand_mean).abs().max(axis=0)

volcano_df = kw_df.copy()
volcano_df["max_abs_diff"] = max_abs_diff[volcano_df["ROI"]].values

fig, ax = plt.subplots(figsize=(7, 5))

sig_mask = volcano_df["significant"].values
nsig_mask = ~sig_mask

ax.scatter(
    volcano_df.loc[nsig_mask, "max_abs_diff"],
    volcano_df.loc[nsig_mask, "-log10p"],
    color="#aaaaaa",
    s=18,
    alpha=0.55,
    edgecolors="none",
    label="Not significant",
    rasterized=True,
)

ax.scatter(
    volcano_df.loc[sig_mask, "max_abs_diff"],
    volcano_df.loc[sig_mask, "-log10p"],
    color="#d73027",
    s=28,
    alpha=0.80,
    edgecolors="white",
    linewidths=0.3,
    label="FDR significant",
    rasterized=True,
    zorder=3,
)

if kw_df["significant"].any():
    fdr_threshold = -np.log10(
        kw_df.loc[kw_df["significant"], "p_value"].max() + 1e-300
    )
    ax.axhline(
        fdr_threshold,
        color="#f4a582",
        linewidth=1.2,
        linestyle="--",
        label=f"FDR={FDR_ALPHA} threshold",
    )

# Label top significant ROIs; if none significant, label top 8 overall.
label_df = volcano_df[sig_mask].head(8)
if label_df.empty:
    label_df = volcano_df.head(8)

for _, row in label_df.iterrows():
    ax.annotate(
        _clean_roi(row["ROI"]),
        xy=(row["max_abs_diff"], row["-log10p"]),
        xytext=(5, 3),
        textcoords="offset points",
        fontsize=6.5,
        color="#333333",
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
    )

ax.set_xlabel("Max |mean z-score − grand mean| across subtypes")
ax.set_ylabel(r"$-\log_{10}(p)$ (Kruskal-Wallis)")
ax.set_title(f"{MODALITY} ROI significance vs effect magnitude")
ax.legend(loc="upper left", framealpha=0.9)
_strip_spines(ax)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"{MODALITY}_volcano.png"), dpi=150)
plt.close(fig)
print("Saved: volcano plot")


# ============================================================
# FIGURE 3 — Cohen's d effect-size bar, K=2 only
# ============================================================
if mwu_df is not None and not mwu_df.empty:
    top_effect = mwu_df.reindex(
        mwu_df["cohen_d"].abs().sort_values(ascending=False).index
    ).head(TOP_K).copy()

    top_effect = top_effect.sort_values("cohen_d")

    colors_bar = [
        "#d73027" if d > 0 else "#4393c3"
        for d in top_effect["cohen_d"]
    ]

    sig_marker = [
        "★" if s else ""
        for s in top_effect["sig_mwu"]
    ]

    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(top_effect) + 1.8))

    ax.barh(
        range(len(top_effect)),
        top_effect["cohen_d"].values,
        color=colors_bar,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.set_yticks(range(len(top_effect)))
    ax.set_yticklabels(
        [_clean_roi(r) for r in top_effect["ROI"]],
        fontsize=8.5,
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Cohen's d  positive = higher in {sub_labels[0]}")
    ax.set_title(f"Top {len(top_effect)} {MODALITY} ROIs by |Cohen's d|")

    for i, (d, star) in enumerate(zip(top_effect["cohen_d"].values, sig_marker)):
        if star:
            offset = 0.03 if d >= 0 else -0.03
            ha = "left" if d >= 0 else "right"
            ax.text(
                d + offset,
                i,
                star,
                va="center",
                ha=ha,
                fontsize=9,
                color="#c0392b",
            )

    patch_pos = mpatches.Patch(
        color="#d73027",
        alpha=0.85,
        label=f"Higher in {sub_labels[0]}",
    )
    patch_neg = mpatches.Patch(
        color="#4393c3",
        alpha=0.85,
        label=f"Higher in {sub_labels[1]}",
    )

    ax.legend(
        handles=[patch_pos, patch_neg],
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )

    _strip_spines(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{MODALITY}_effect_size_bar.png"), dpi=150)
    plt.close(fig)
    print("Saved: effect-size bar")
else:
    print("Skipping effect-size bar: n_sub != 2 or MWU results are empty.")


# ============================================================
# FIGURE 4 — Box + strip plots for top ROIs
# ============================================================
n_top_plot = min(TOP_K, len(top_rois))

if n_top_plot > 0:
    ncols = 5
    nrows = int(np.ceil(n_top_plot / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.5, nrows * 3.4),
    )

    axes = np.array(axes).reshape(-1)

    for i, roi in enumerate(top_rois[:n_top_plot]):
        ax = axes[i]

        data_by_sub = [
            df_z.loc[df_z[SUBTYPE_COL] == k, roi].dropna().values
            for k in subtypes
        ]

        bp = ax.boxplot(
            data_by_sub,
            patch_artist=True,
            notch=False,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.6),
            whiskerprops=dict(linewidth=0.9, color="#555555"),
            capprops=dict(linewidth=0.9, color="#555555"),
            flierprops=dict(
                marker="o",
                markersize=2,
                alpha=0.3,
                markerfacecolor="#888888",
                markeredgecolor="none",
            ),
        )

        for patch, col in zip(bp["boxes"], sub_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
            patch.set_linewidth(0.6)

        for j, (vals, col) in enumerate(zip(data_by_sub, sub_colors)):
            rng = np.random.default_rng(99 + i + j)
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(
                np.full(len(vals), j + 1) + jitter,
                vals,
                color=col,
                alpha=0.32,
                s=9,
                edgecolors="none",
                zorder=3,
            )

        row = kw_df[kw_df["ROI"] == roi]

        if not row.empty:
            p_fdr = row["p_corrected"].values[0]
            stars = _sig_stars(p_fdr)
            title = f"{_clean_roi(roi)}\n(FDR {stars}, p={p_fdr:.3g})"
        else:
            title = _clean_roi(roi)

        ax.set_title(title, fontsize=8, pad=5)
        ax.set_xticks(range(1, n_sub + 1))
        ax.set_xticklabels(sub_labels, fontsize=7.5)
        ax.set_ylabel("z-score", fontsize=8)
        _strip_spines(ax)

    for j in range(n_top_plot, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Top {n_top_plot} {MODALITY} ROIs — subtype comparison",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{MODALITY}_top_roi_boxplots.png"), dpi=150)
    plt.close(fig)
    print("Saved: top ROI boxplots")
else:
    print("Skipping top ROI boxplots: no top ROIs available.")


# ============================================================
# FIGURE 5 — Pairwise ROI mean difference matrix, K>2 only
# ============================================================
if n_sub > 2:
    top20 = kw_df["ROI"].values[:20]
    clean20 = [_clean_roi(r) for r in top20]

    pairs = list(combinations(range(n_sub), 2))
    n_pairs = len(pairs)

    pair_labels = [
        f"{sub_labels[a]}−{sub_labels[b]}"
        for a, b in pairs
    ]

    diff_matrix = np.zeros((len(top20), n_pairs))

    for pi, (a, b) in enumerate(pairs):
        ka, kb = subtypes[a], subtypes[b]
        ma = roi_mean.loc[ka, top20].values
        mb = roi_mean.loc[kb, top20].values
        diff_matrix[:, pi] = ma - mb

    fig, ax = plt.subplots(
        figsize=(max(5, n_pairs * 2.2), max(5, 0.4 * len(top20) + 1.5))
    )

    vmax = np.abs(diff_matrix).max()
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0

    im = ax.imshow(
        diff_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(pair_labels, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(clean20, fontsize=8)
    ax.set_title(f"Pairwise mean differences top 20 {MODALITY} ROIs", pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Mean z-score difference", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{MODALITY}_pairwise_diff_heatmap.png"), dpi=150)
    plt.close(fig)
    print("Saved: pairwise diff heatmap")
else:
    print("Skipping pairwise diff heatmap: only 2 subtypes.")


# ============================================================
# Done
# ============================================================
print(f"\nAll ROI analysis outputs saved to: {OUTPUT_DIR}")
print(f"Significant ROIs (FDR<{FDR_ALPHA}): {kw_df['significant'].sum()} / {len(kw_df)}")

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from scipy.stats import f_oneway, ttest_ind
# from sklearn.preprocessing import StandardScaler
# from statsmodels.stats.multitest import multipletests


# MRI_PATH = "./dataset/PET_whole.csv"
# CHAR_PATH = "./results/subtype_analysis/subtype_with_bestK.csv"
# OUTPUT_DIR = "./results/subtype_analysis"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# PTID_COL = "PTID"
# SUBTYPE_COL = "subtype"  

# mri_df = pd.read_csv(MRI_PATH)
# char_df = pd.read_csv(CHAR_PATH)

# print("MRI shape:", mri_df.shape)
# print("CHAR shape:", char_df.shape)

# mri_df[PTID_COL] = mri_df[PTID_COL].astype(str).str.strip()
# char_df[PTID_COL] = char_df[PTID_COL].astype(str).str.strip()


# print("Characteristic columns:", char_df.columns.tolist())

# df = mri_df.merge(
#     char_df[[PTID_COL, SUBTYPE_COL]],
#     on=PTID_COL,
#     how='inner'
# )

# print("After merge shape:", df.shape)

# roi_cols = [
#     c for c in df.columns
#     if ('suvr' in c.lower())
# ]

# print(f"Number of ROI features: {len(roi_cols)}")
# print("Example ROI:", roi_cols[:10])

# df = df.dropna(subset=roi_cols + [SUBTYPE_COL])

# print("After dropna:", df.shape)

# scaler = StandardScaler()
# df[roi_cols] = scaler.fit_transform(df[roi_cols])

# print("ROI standardized")

# subtypes = sorted(df[SUBTYPE_COL].unique())
# print("Subtypes:", subtypes)

# roi_mean = df.groupby(SUBTYPE_COL)[roi_cols].mean()
# roi_mean.to_csv(os.path.join(OUTPUT_DIR, "PET_roi_mean_by_subtype.csv"))

# print("ROI mean saved")

# anova_results = {}

# for roi in roi_cols:
#     groups = [df[df[SUBTYPE_COL] == k][roi] for k in subtypes]

#     if any(len(g) == 0 for g in groups):
#         continue

#     stat, p = f_oneway(*groups)
#     anova_results[roi] = p

# anova_df = pd.DataFrame.from_dict(anova_results, orient='index', columns=['p_value'])
# anova_df = anova_df.sort_values('p_value')


# reject, pvals_corrected, _, _ = multipletests(anova_df['p_value'], method='fdr_bh')
# anova_df['p_corrected'] = pvals_corrected
# anova_df['significant'] = reject

# anova_df.to_csv(os.path.join(OUTPUT_DIR, "PET_anova_results.csv"))
# print("ANOVA done")

# if len(subtypes) == 2:
#     print("Running t-test")

#     g0 = df[df[SUBTYPE_COL] == subtypes[0]]
#     g1 = df[df[SUBTYPE_COL] == subtypes[1]]

#     ttest_results = {}

#     for roi in roi_cols:
#         stat, p = ttest_ind(g0[roi], g1[roi])
#         ttest_results[roi] = p

#     ttest_df = pd.DataFrame.from_dict(ttest_results, orient='index', columns=['p_value'])
#     ttest_df = ttest_df.sort_values('p_value')

#     reject, pvals_corrected, _, _ = multipletests(ttest_df['p_value'], method='fdr_bh')
#     ttest_df['p_corrected'] = pvals_corrected
#     ttest_df['significant'] = reject

#     ttest_df.to_csv(os.path.join(OUTPUT_DIR, "PET_ttest_results.csv"))

#     print("T-test done")

# plt.figure(figsize=(12, 6))
# sns.heatmap(roi_mean, cmap='coolwarm', center=0)
# plt.title("ROI Mean per Subtype")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "PET_heatmap.png"))
# plt.close()

# print("Heatmap saved")

# TOP_K = 10
# top_rois = anova_df.head(TOP_K).index.tolist()

# for roi in top_rois:
#     plt.figure()
#     sns.boxplot(x=SUBTYPE_COL, y=roi, data=df)
#     plt.title(roi)
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, f"PET_{roi}.png"))
#     plt.close()

# print("Boxplots saved")


# def cohen_d(x, y):
#     return (np.mean(x) - np.mean(y)) / np.sqrt(
#         (np.std(x)**2 + np.std(y)**2) / 2
#     )

# if len(subtypes) == 2:
#     effect_sizes = {}

#     for roi in roi_cols:
#         effect_sizes[roi] = cohen_d(g0[roi], g1[roi])

#     effect_df = pd.DataFrame.from_dict(effect_sizes, orient='index', columns=['cohen_d'])
#     effect_df = effect_df.reindex(effect_df['cohen_d'].abs().sort_values(ascending=False).index)

#     effect_df.to_csv(os.path.join(OUTPUT_DIR, "PET_effect_size.csv"))

#     print("Effect size saved")

# print("\n Top significant ROIs:")
# print(anova_df[anova_df['significant']].head(20))
