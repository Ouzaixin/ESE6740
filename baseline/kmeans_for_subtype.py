import os
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from scipy import stats

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE

from model import AE
from dataset import ROIDataset

matplotlib.use("Agg")

# ============================================================
# Publication-quality plot style
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
    "figure.constrained_layout.use": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Colour palettes
SUBTYPE_CMAP = "tab10"
DX_COLORS = {"CN": "#2166ac", "AD": "#d6604d"}   # blue / red
AGE_CMAP   = "plasma"
APOE_COLORS = {0: "#4dac26", 1: "#f1b6da", 2: "#d01c8b"}  # 0 / 1 / 2 alleles

POINT_SIZE   = 28
POINT_ALPHA  = 0.82
EDGE_WIDTH   = 0.25
EDGE_COLOR   = "white"

# ============================================================
# Paths
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path      = "./results/best.pth"
preprocess_path = "./results/preprocess_objects.pkl"
save_dir        = "./results/subtype_analysis"
os.makedirs(save_dir, exist_ok=True)


# ============================================================
# Load data
# ============================================================
mri_df = pd.read_csv("./dataset/MRI_CN_AD.csv")
pet_df = pd.read_csv("./dataset/PET_CN_AD.csv")

assert "PTID" in mri_df.columns
assert "PTID" in pet_df.columns

full_mri = mri_df.reset_index(drop=True)
full_pet = pet_df.reset_index(drop=True)


# ============================================================
# Load preprocessing objects
# ============================================================
with open(preprocess_path, "rb") as f:
    preprocess_objects = pickle.load(f)

mri_imputer = preprocess_objects["mri_imputer"]
pet_imputer = preprocess_objects["pet_imputer"]
mri_scaler  = preprocess_objects["mri_scaler"]
pet_scaler  = preprocess_objects["pet_scaler"]
label_encoder = preprocess_objects["label_encoder"]
age_mean    = preprocess_objects["age_mean"]
age_std     = preprocess_objects["age_std"]
apoe_fill   = preprocess_objects.get("apoe_fill", 0)


# ============================================================
# Build full dataset
# ============================================================
full_dataset = ROIDataset(
    full_mri, full_pet,
    mri_imputer=mri_imputer, pet_imputer=pet_imputer,
    mri_scaler=mri_scaler,   pet_scaler=pet_scaler,
    label_encoder=label_encoder,
    age_mean=age_mean, age_std=age_std, apoe_fill=apoe_fill,
    fit=False,
)
full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)


# ============================================================
# Load model
# ============================================================
num_rois    = full_dataset.X_mri.shape[1]
latent      = 128
subtype_dim = latent // 16

model = AE(
    num_rois=num_rois, latent=latent, dim=512,
    beta_mi=1.0, beta_ib=2.0,
    subtype_dim=subtype_dim, dropout_p=0.0,
    num_classes=len(label_encoder.classes_),
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print(f"Model loaded | num_rois={num_rois} | subtype_dim={subtype_dim}")
print(f"Classes: {list(label_encoder.classes_)}")


# ============================================================
# Extract z_sub
# ============================================================
all_z    = []
all_y    = []
all_apoe = []
all_age  = []

with torch.no_grad():
    for mri, pet, y, apoe, age in full_loader:
        out = model(mri.to(device), pet.to(device), y.to(device))
        all_z.append(out["z_sub"].cpu().numpy())
        all_y.append(y.cpu().numpy())
        all_apoe.append(apoe.numpy())
        all_age.append(age.numpy())

all_z    = np.concatenate(all_z, axis=0)
all_y    = np.concatenate(all_y, axis=0)
all_apoe = np.concatenate(all_apoe, axis=0)
all_age  = np.concatenate(all_age, axis=0)

all_dx   = label_encoder.inverse_transform(all_y)           # e.g. "CN" / "AD"
all_age_raw = all_age * age_std + age_mean                  # recover real age

print(f"z_sub shape: {all_z.shape}")


# ============================================================
# Dimensionality reduction helpers
# ============================================================
def fit_pca_2d(Z):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(Z), pca

def fit_tsne_2d(Z, perp=None):
    n = Z.shape[0]
    perp = perp or min(30, max(5, (n - 1) // 3))
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=perp, init="pca", learning_rate="auto")
    return tsne.fit_transform(Z)


# ============================================================
# Scatter helper
# ============================================================
def _scatter_base(ax, xy, c, cmap=None, norm=None, s=POINT_SIZE,
                  alpha=POINT_ALPHA, edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH,
                  vmin=None, vmax=None):
    return ax.scatter(
        xy[:, 0], xy[:, 1],
        c=c, cmap=cmap, norm=norm,
        s=s, alpha=alpha,
        edgecolors=edgecolors, linewidths=linewidths,
        vmin=vmin, vmax=vmax,
        rasterized=True,
    )

def _axis_style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, pad=10)
    ax.tick_params(length=3, width=0.7)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)


# ============================================================
# KMeans selection — full cohort
# ============================================================
n_samples = all_z.shape[0]
max_k = min(10, n_samples - 1)
Ks = list(range(2, max_k + 1))

sil_scores = []
inertias   = []
best_k, best_ss, best_labels = None, -1, None

for K in Ks:
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    lbs = km.fit_predict(all_z)
    ss  = silhouette_score(all_z, lbs)
    sil_scores.append(ss)
    inertias.append(km.inertia_)
    print(f"  K={K} | SS={ss:.4f} | Inertia={km.inertia_:.1f}")
    if ss > best_ss:
        best_ss, best_k, best_labels = ss, K, lbs

print(f"\nBest K={best_k}  SS={best_ss:.4f}\n")

# Final fit at best K
final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
subtypes = final_km.fit_predict(all_z)


# ============================================================
# Build result_df (full cohort)
# ============================================================
result_df = pd.DataFrame({
    "PTID":          full_mri["PTID"].astype(str).values,
    "subtype":       subtypes,
    "DX_encoded":    all_y,
    "DX_label":      all_dx,
    "APOE4":         all_apoe,
    "AGE_z":         all_age,
    "AGE_recovered": all_age_raw,
})

result_df.to_csv(os.path.join(save_dir, "subtype_with_bestK.csv"), index=False)


# ============================================================
# Dimensionality reduction — full cohort
# ============================================================
z_pca, pca_obj = fit_pca_2d(all_z)
z_tsne          = fit_tsne_2d(all_z)

result_df["pca_1"]  = z_pca[:, 0]
result_df["pca_2"]  = z_pca[:, 1]
result_df["tsne_1"] = z_tsne[:, 0]
result_df["tsne_2"] = z_tsne[:, 1]


# ============================================================
# ---- SECTION A: Global analysis plots ----------------------
# ============================================================
print("Generating global analysis plots …")
glob_dir = os.path.join(save_dir, "global")
os.makedirs(glob_dir, exist_ok=True)

def _k_selection_panel(Ks, sil_scores, inertias, save_path, title_prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    ax = axes[0]
    ax.plot(Ks, sil_scores, marker="o", color="#2c7bb6", lw=1.8, ms=6, zorder=3)
    best_idx = int(np.argmax(sil_scores))
    ax.scatter([Ks[best_idx]], [sil_scores[best_idx]],
               color="#d73027", s=80, zorder=4, label=f"Best K={Ks[best_idx]}")
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"{title_prefix}Silhouette vs K")
    ax.legend(framealpha=0.9)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

    ax = axes[1]
    ax.plot(Ks, inertias, marker="o", color="#4dac26", lw=1.8, ms=6)
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Inertia (within-cluster SS)")
    ax.set_title(f"{title_prefix}Elbow curve")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

    fig.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

_k_selection_panel(
    Ks, sil_scores, inertias,
    save_path=os.path.join(glob_dir, "k_selection.png"),
    title_prefix="Global — ",
)


def _global_embedding_grid(
    xy,
    labels_subtype,
    dx_arr,
    age_arr,
    apoe_arr,
    method_name,
    save_path,
):
    """
    4-panel figure (2×2 layout):
        top-left:  colored by subtype
        top-right: CN same color, AD colored by APOE4
        bottom-left: AD same color, CN colored by APOE4
        bottom-right: colored by age
    """
    unique_subtypes = np.unique(labels_subtype)
    cmap_sub = plt.get_cmap(SUBTYPE_CMAP)
    sub_colors = {
        k: cmap_sub(i / max(len(unique_subtypes) - 1, 1))
        for i, k in enumerate(unique_subtypes)
    }

    dx_arr = np.asarray(dx_arr).astype(str)
    apoe_arr = np.asarray(apoe_arr).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ------------------------------------------------------------
    # Panel 1 (top-left): subtype
    # ------------------------------------------------------------
    ax = axes[0, 0]
    for k in unique_subtypes:
        mask = labels_subtype == k
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            color=sub_colors[k],
            label=f"Subtype {k}",
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            rasterized=True,
        )
    ax.legend(title="Subtype", title_fontsize=8, loc="best", markerscale=1.2)
    _axis_style(ax, f"{method_name} 1", f"{method_name} 2", "Subtype clusters")

    # ------------------------------------------------------------
    # Panel 2 (top-right): CN fixed, AD colored by APOE4
    # ------------------------------------------------------------
    ax = axes[0, 1]

    cn_mask = dx_arr == "CN"
    ad_mask = dx_arr == "AD"

    # Plot CN first in one neutral/diagnosis color
    ax.scatter(
        xy[cn_mask, 0],
        xy[cn_mask, 1],
        color=DX_COLORS.get("CN", "#2166ac"),
        label="CN",
        s=POINT_SIZE,
        alpha=0.45,
        edgecolors=EDGE_COLOR,
        linewidths=EDGE_WIDTH,
        rasterized=True,
    )

    # Plot AD split by APOE4 count
    for av in [0, 1, 2]:
        mask = ad_mask & (apoe_arr == av)
        if mask.sum() == 0:
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            color=APOE_COLORS.get(av, f"C{av}"),
            label=f"AD, APOE4={av}",
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            rasterized=True,
        )
    ax.legend(title="Diagnosis/APOE4", title_fontsize=8, loc="best", markerscale=1.2)
    _axis_style(ax, f"{method_name} 1", f"{method_name} 2", "CN fixed; AD by APOE4")

    # ------------------------------------------------------------
    # Panel 3 (bottom-left): AD fixed, CN colored by APOE4
    # ------------------------------------------------------------
    ax = axes[1, 0]

    # Plot AD first in one fixed color
    ax.scatter(
        xy[ad_mask, 0],
        xy[ad_mask, 1],
        color=DX_COLORS.get("AD", "#d6604d"),
        label="AD",
        s=POINT_SIZE,
        alpha=0.45,
        edgecolors=EDGE_COLOR,
        linewidths=EDGE_WIDTH,
        rasterized=True,
    )

    # Plot CN split by APOE4 count
    for av in [0, 1, 2]:
        mask = cn_mask & (apoe_arr == av)
        if mask.sum() == 0:
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            color=APOE_COLORS.get(av, f"C{av}"),
            label=f"CN, APOE4={av}",
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            rasterized=True,
        )
    ax.legend(title="Diagnosis/APOE4", title_fontsize=8, loc="best", markerscale=1.2)
    _axis_style(ax, f"{method_name} 1", f"{method_name} 2", "AD fixed; CN by APOE4")

    # ------------------------------------------------------------
    # Panel 4 (bottom-right): Age
    # ------------------------------------------------------------
    ax = axes[1, 1]
    norm_age = Normalize(vmin=age_arr.min(), vmax=age_arr.max())
    sc = _scatter_base(ax, xy, c=age_arr, cmap=AGE_CMAP, norm=norm_age)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Age (years)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    _axis_style(ax, f"{method_name} 1", f"{method_name} 2", "Age")

    fig.suptitle(
        f"Global cohort — {method_name} of $z_s$",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


_global_embedding_grid(
    z_tsne,
    subtypes,
    all_dx,
    all_age_raw,
    all_apoe,
    method_name="t-SNE",
    save_path=os.path.join(glob_dir, "global_tsne_grid.png"),
)

_global_embedding_grid(
    z_pca,
    subtypes,
    all_dx,
    all_age_raw,
    all_apoe,
    method_name="PCA",
    save_path=os.path.join(glob_dir, "global_pca_grid.png"),
)

# PCA variance explained bar chart
var_exp = pca_obj.explained_variance_ratio_ * 100
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.bar(["PC 1", "PC 2"], var_exp, color=["#4393c3", "#92c5de"], edgecolor="white", linewidth=0.5)
for i, v in enumerate(var_exp):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Variance explained (%)")
ax.set_title("PCA variance explained")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(glob_dir, "pca_variance_explained.png"), dpi=150)
plt.close(fig)


# Global subtype × DX stacked bar
dx_pct_global = pd.crosstab(result_df["subtype"], result_df["DX_label"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(max(4, best_k * 1.4), 3.8))
bottom = np.zeros(best_k)
dx_labels = dx_pct_global.columns.tolist()
bar_colors = [DX_COLORS.get(d, f"C{i}") for i, d in enumerate(dx_labels)]
for dx, col in zip(dx_labels, bar_colors):
    vals = dx_pct_global[dx].values
    ax.bar(range(best_k), vals, bottom=bottom, color=col, label=dx, edgecolor="white", linewidth=0.5)
    bottom += vals
ax.set_xticks(range(best_k))
ax.set_xticklabels([f"Subtype {k}" for k in range(best_k)])
ax.set_ylabel("Proportion (%)")
ax.set_title("Diagnosis composition per subtype")
ax.legend(title="Diagnosis", bbox_to_anchor=(1.01, 1), loc="upper left")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(glob_dir, "global_subtype_dx_stacked.png"), dpi=150)
plt.close(fig)

print(f"  Saved global plots → {glob_dir}")


# ============================================================
# ---- SECTION B: AD-only analysis ---------------------------
# ============================================================

def run_kmeans_ad_only(
    all_z, result_df, save_dir,
    dx_col="DX_label", ad_label="AD",
    k_min=2, k_max=8,
    random_state=42,
):
    ad_save_dir = os.path.join(save_dir, "AD_only")
    os.makedirs(ad_save_dir, exist_ok=True)

    ad_mask  = result_df[dx_col].astype(str).values == ad_label
    z_ad     = all_z[ad_mask]
    result_ad = result_df.loc[ad_mask].copy().reset_index(drop=True)

    n_ad = len(result_ad)
    print(f"\nAD-only clustering | n={n_ad}")

    if n_ad < k_min + 1:
        raise ValueError(f"Too few AD subjects for clustering: n={n_ad}")

    # ---- K selection ----
    max_k_ad = min(k_max, n_ad - 1)
    Ks_ad    = list(range(k_min, max_k_ad + 1))

    sil_ad, iner_ad = [], []
    best_k_ad, best_ss_ad, best_labels_ad = None, -1, None

    for K in Ks_ad:
        km  = KMeans(n_clusters=K, random_state=random_state, n_init=10)
        lbs = km.fit_predict(z_ad)
        ss  = silhouette_score(z_ad, lbs)
        sil_ad.append(ss)
        iner_ad.append(km.inertia_)
        print(f"  AD K={K} | SS={ss:.4f} | Inertia={km.inertia_:.1f}")
        if ss > best_ss_ad:
            best_ss_ad, best_k_ad, best_labels_ad = ss, K, lbs

    print(f"\n  Best AD K={best_k_ad}  SS={best_ss_ad:.4f}")

    _k_selection_panel(
        Ks_ad, sil_ad, iner_ad,
        save_path=os.path.join(ad_save_dir, "AD_k_selection.png"),
        title_prefix="AD — ",
    )

    # Final fit
    final_km_ad = KMeans(n_clusters=best_k_ad, random_state=random_state, n_init=10)
    ad_subtypes = final_km_ad.fit_predict(z_ad)
    final_ss_ad = silhouette_score(z_ad, ad_subtypes)
    result_ad["AD_subtype"] = ad_subtypes

    # ---- Dimensionality reduction ----
    z_ad_pca, pca_ad = fit_pca_2d(z_ad)
    z_ad_tsne        = fit_tsne_2d(z_ad)

    result_ad["pca_1"]  = z_ad_pca[:, 0]
    result_ad["pca_2"]  = z_ad_pca[:, 1]
    result_ad["tsne_1"] = z_ad_tsne[:, 0]
    result_ad["tsne_2"] = z_ad_tsne[:, 1]

    # ---- Plot helper for AD ----
    def _ad_embedding_grid(xy, method_name, save_path):
        """
        4-panel figure (2×2 layout):
            top-left:     AD subtype
            top-right:    APOE4
            bottom-left:  Age
            bottom-right: raw z_s (dim 0 vs dim 1)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: AD subtype
        ax = axes[0, 0]
        cmap_sub = plt.get_cmap(SUBTYPE_CMAP)
        unique_sub = np.unique(ad_subtypes)
        for k in unique_sub:
            mask = ad_subtypes == k
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       color=cmap_sub(k / max(len(unique_sub) - 1, 1)),
                       label=f"AD-{k}", s=POINT_SIZE, alpha=POINT_ALPHA,
                       edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH, rasterized=True)
        ax.legend(title="AD subtype", title_fontsize=8, loc="best", markerscale=1.2)
        _axis_style(ax, f"{method_name} 1", f"{method_name} 2", "AD subtype")

        # Panel 2: APOE4
        ax = axes[0, 1]
        apoe_vals = result_ad["APOE4"].astype(int).values
        unique_apoe = np.unique(apoe_vals)
        for av in unique_apoe:
            mask = apoe_vals == av
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       color=APOE_COLORS.get(av, f"C{av}"),
                       label=f"APOE4={av}", s=POINT_SIZE, alpha=POINT_ALPHA,
                       edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH, rasterized=True)
        ax.legend(title="APOE4 alleles", title_fontsize=8, loc="best", markerscale=1.2)
        _axis_style(ax, f"{method_name} 1", f"{method_name} 2", "APOE4 genotype")

        # Panel 3: Age
        ax = axes[1, 0]
        age_vals = result_ad["AGE_recovered"].values
        norm_age = Normalize(vmin=age_vals.min(), vmax=age_vals.max())
        sc = _scatter_base(ax, xy, c=age_vals, cmap=AGE_CMAP, norm=norm_age)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Age (years)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        _axis_style(ax, f"{method_name} 1", f"{method_name} 2", "Age")

        # Panel 4: z_s dim 0 vs dim 1 raw (no reduction)
        ax = axes[1, 1]
        for k in unique_sub:
            mask = ad_subtypes == k
            ax.scatter(z_ad[mask, 0], z_ad[mask, 1],
                       color=cmap_sub(k / max(len(unique_sub) - 1, 1)),
                       label=f"AD-{k}", s=POINT_SIZE, alpha=POINT_ALPHA,
                       edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH, rasterized=True)
        ax.legend(title="AD subtype", title_fontsize=8, loc="best", markerscale=1.2)
        _axis_style(ax, r"$z_s$ dim 0", r"$z_s$ dim 1", r"Raw $z_s$ (dim 0 vs 1)")

        fig.suptitle(f"AD-only analysis — {method_name} of $z_s$",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout(pad=2.0)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    _ad_embedding_grid(z_ad_tsne, "t-SNE",
                       os.path.join(ad_save_dir, "AD_tsne_grid.png"))
    _ad_embedding_grid(z_ad_pca,  "PCA",
                       os.path.join(ad_save_dir, "AD_pca_grid.png"))

    # PCA variance explained (AD)
    var_ad = pca_ad.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(["PC 1", "PC 2"], var_ad, color=["#d73027", "#f4a582"],
           edgecolor="white", linewidth=0.5)
    for i, v in enumerate(var_ad):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("AD PCA variance explained")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(ad_save_dir, "AD_pca_variance_explained.png"), dpi=150)
    plt.close(fig)


    # ---- Refined AD subtype characterization ----

    # 1. Box plots: Age per subtype
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))

    cmap_sub = plt.get_cmap(SUBTYPE_CMAP)
    unique_sub = np.sort(np.unique(ad_subtypes))

    ax = axes[0]
    age_by_sub = [result_ad.loc[result_ad["AD_subtype"] == k, "AGE_recovered"].values
                  for k in unique_sub]
    bp = ax.boxplot(age_by_sub, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, k in zip(bp["boxes"], unique_sub):
        patch.set_facecolor(cmap_sub(k / max(len(unique_sub) - 1, 1)))
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(unique_sub) + 1))
    ax.set_xticklabels([f"AD-{k}" for k in unique_sub])
    ax.set_ylabel("Age (years)")
    ax.set_title("Age distribution by AD subtype")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

    # 2. APOE4 grouped bar per subtype
    ax = axes[1]
    apoe_counts = pd.crosstab(result_ad["AD_subtype"], result_ad["APOE4"])
    apoe_pct    = apoe_counts.div(apoe_counts.sum(axis=1), axis=0) * 100
    apoe_cols_present = apoe_pct.columns.tolist()

    x       = np.arange(len(unique_sub))
    n_apoe  = len(apoe_cols_present)
    width   = 0.7 / n_apoe

    for i, av in enumerate(apoe_cols_present):
        offset = (i - (n_apoe - 1) / 2) * width
        bars = ax.bar(x + offset, apoe_pct[av].values,
                      width=width,
                      color=APOE_COLORS.get(int(av), f"C{i}"),
                      label=f"APOE4={av}",
                      edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"AD-{k}" for k in unique_sub])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("APOE4 distribution by AD subtype")
    ax.legend(title="APOE4 alleles", title_fontsize=8)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

    fig.suptitle("AD subtype clinical characterization",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(ad_save_dir, "AD_subtype_clinical_boxplot.png"), dpi=150)
    plt.close(fig)


    # 3. z_s dimension heatmap: mean per subtype  (all dims)
    z_sub_means = np.array([z_ad[ad_subtypes == k].mean(axis=0)
                             for k in unique_sub])   # [K, z_dim]

    fig, ax = plt.subplots(figsize=(max(7, subtype_dim * 0.7), 3.0 + 0.35 * best_k_ad))
    im = ax.imshow(z_sub_means, aspect="auto", cmap="RdBu_r",
                   norm=Normalize(vmin=-np.abs(z_sub_means).max(),
                                  vmax= np.abs(z_sub_means).max()))
    ax.set_xticks(range(subtype_dim))
    ax.set_xticklabels([f"$z_{{{i}}}$" for i in range(subtype_dim)], fontsize=8)
    ax.set_yticks(range(len(unique_sub)))
    ax.set_yticklabels([f"AD-{k}" for k in unique_sub])
    ax.set_xlabel(r"$z_s$ dimension")
    ax.set_title(r"Mean $z_s$ per AD subtype")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Mean value", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ad_save_dir, "AD_subtype_zsub_heatmap.png"), dpi=150)
    plt.close(fig)


    # 4. Statistical tests: age and APOE4 across AD subtypes
    print("\n=== Statistical tests (AD subtypes) ===")

    # Age: Kruskal-Wallis (nonparametric, safe for small groups)
    age_groups = [result_ad.loc[result_ad["AD_subtype"] == k, "AGE_recovered"].values
                  for k in unique_sub]
    if len(unique_sub) > 1:
        kw_stat, kw_p = stats.kruskal(*age_groups)
        print(f"  Age (Kruskal-Wallis): H={kw_stat:.3f}, p={kw_p:.4f}")
    else:
        kw_stat, kw_p = np.nan, np.nan
        print("  Age: only 1 subtype, skipping test.")

    # APOE4: Chi-squared
    apoe_contingency = pd.crosstab(result_ad["AD_subtype"], result_ad["APOE4"])
    if apoe_contingency.shape[0] > 1 and apoe_contingency.shape[1] > 1:
        chi2, chi2_p, dof, _ = stats.chi2_contingency(apoe_contingency.values)
        print(f"  APOE4 (Chi-squared): chi2={chi2:.3f}, p={chi2_p:.4f}, df={dof}")
    else:
        chi2, chi2_p, dof = np.nan, np.nan, np.nan
        print("  APOE4: insufficient categories for chi2.")

    stats_df = pd.DataFrame({
        "test": ["Kruskal-Wallis (age)", "Chi-squared (APOE4)"],
        "statistic": [kw_stat, chi2],
        "p_value": [kw_p, chi2_p],
        "df": [np.nan, dof],
    })
    stats_df.to_csv(os.path.join(ad_save_dir, "AD_subtype_stats.csv"), index=False)


    # ---- Summary tables ----
    ad_summary = result_ad.groupby("AD_subtype")[["AGE_recovered", "APOE4"]].agg(
        AGE_mean=("AGE_recovered", "mean"),
        AGE_std=("AGE_recovered", "std"),
        AGE_median=("AGE_recovered", "median"),
        N=("AGE_recovered", "count"),
        APOE4_mean=("APOE4", "mean"),
        APOE4_std=("APOE4", "std"),
    )
    ad_summary.to_csv(os.path.join(ad_save_dir, "AD_subtype_summary.csv"))
    print("\n=== AD subtype summary ===")
    print(ad_summary.to_string())

    apoe_pct.to_csv(os.path.join(ad_save_dir, "AD_APOE4_percent.csv"))
    apoe_counts.to_csv(os.path.join(ad_save_dir, "AD_APOE4_count.csv"))

    result_ad.to_csv(os.path.join(ad_save_dir, "AD_subtype_with_bestK.csv"), index=False)

    print(f"\n  Final AD K={best_k_ad} | SS={final_ss_ad:.4f}")
    print(f"  Saved AD-only results → {ad_save_dir}")

    return result_ad, best_k_ad, final_ss_ad


result_ad, best_k_ad, final_ss_ad = run_kmeans_ad_only(
    all_z=all_z,
    result_df=result_df,
    save_dir=save_dir,
    dx_col="DX_label",
    ad_label="AD",
    k_min=2,
    k_max=8,
)


# ============================================================
# Compare global subtypes against DX_bl (characteristic.csv)
# ============================================================
char_path = "./dataset/characteristic.csv"
if os.path.exists(char_path):
    char_df = pd.read_csv(char_path)
    char_df["PTID"] = char_df["PTID"].astype(str)

    merged = pd.merge(result_df, char_df, on="PTID", how="inner")
    merged = merged.dropna(subset=["DX_bl"])
    merged["DX_label_factorized"] = pd.factorize(merged["DX_bl"])[0]

    ari = adjusted_rand_score(merged["DX_label_factorized"], merged["subtype"])

    dx_count = pd.crosstab(merged["subtype"], merged["DX_bl"])
    dx_pct   = pd.crosstab(merged["subtype"], merged["DX_bl"], normalize="index") * 100

    dx_count.to_csv(os.path.join(save_dir, "subtype_dx_count.csv"))
    dx_pct.to_csv(os.path.join(save_dir, "subtype_dx_percent.csv"))

    print("\n=== Subtype vs DX_bl (count) ===")
    print(dx_count.to_string())
    print("\n=== Subtype vs DX_bl (%) ===")
    print(dx_pct.round(1).to_string())
    print(f"\nARI (subtype vs DX_bl) = {ari:.4f}")
else:
    print(f"\ncharacteristic.csv not found at {char_path}; skipping ARI computation.")


# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 52)
print(f"Global  — Best K = {best_k}   | SS = {best_ss:.4f}")
print(f"AD-only — Best K = {best_k_ad} | SS = {final_ss_ad:.4f}")
print(f"All outputs saved to: {save_dir}")
print("=" * 52)
# import os
# import pickle

# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, adjusted_rand_score
# from sklearn.manifold import TSNE

# from model import AE
# from dataset import ROIDataset


# device = "cuda" if torch.cuda.is_available() else "cpu"

# model_path = "./results/best.pth"
# preprocess_path = "./results/preprocess_objects.pkl"
# save_dir = "./results/subtype_analysis"
# os.makedirs(save_dir, exist_ok=True)


# # ------------------------------------------------------------
# # Load data
# # ------------------------------------------------------------
# mri_df = pd.read_csv("./dataset/MRI_CN_AD.csv")
# pet_df = pd.read_csv("./dataset/PET_CN_AD.csv")

# assert "PTID" in mri_df.columns
# assert "PTID" in pet_df.columns

# full_mri = mri_df.reset_index(drop=True)
# full_pet = pet_df.reset_index(drop=True)


# # ------------------------------------------------------------
# # Load train-time preprocessing objects
# # ------------------------------------------------------------
# with open(preprocess_path, "rb") as f:
#     preprocess_objects = pickle.load(f)

# mri_imputer = preprocess_objects["mri_imputer"]
# pet_imputer = preprocess_objects["pet_imputer"]
# mri_scaler = preprocess_objects["mri_scaler"]
# pet_scaler = preprocess_objects["pet_scaler"]
# label_encoder = preprocess_objects["label_encoder"]
# age_mean = preprocess_objects["age_mean"]
# age_std = preprocess_objects["age_std"]
# apoe_fill = preprocess_objects.get("apoe_fill", 0)


# # ------------------------------------------------------------
# # Build full dataset using train-time preprocessing
# # ------------------------------------------------------------
# full_dataset = ROIDataset(
#     full_mri,
#     full_pet,
#     mri_imputer=mri_imputer,
#     pet_imputer=pet_imputer,
#     mri_scaler=mri_scaler,
#     pet_scaler=pet_scaler,
#     label_encoder=label_encoder,
#     age_mean=age_mean,
#     age_std=age_std,
#     apoe_fill=apoe_fill,
#     fit=False,
# )

# full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)


# # ------------------------------------------------------------
# # Load trained model
# # ------------------------------------------------------------
# num_rois = full_dataset.X_mri.shape[1]
# latent = 128
# # subtype_dim = latent // 32
# subtype_dim = latent // 16

# model = AE(
#     num_rois=num_rois,
#     latent=latent,
#     dim=512,
#     beta_mi=1.0,
#     beta_ib=2.0,
#     subtype_dim=subtype_dim,
#     dropout_p=0.0,
#     num_classes=len(label_encoder.classes_),
# )

# model.load_state_dict(torch.load(model_path, map_location=device))
# model.to(device)
# model.eval()

# print("Model loaded.")
# print(f"num_rois = {num_rois}")
# print(f"subtype_dim = {subtype_dim}")
# print(f"classes = {list(label_encoder.classes_)}")

# # ------------------------------------------------------------
# # Extract z_sub for all subjects
# # ------------------------------------------------------------
# all_z = []
# all_y = []
# all_apoe = []
# all_age = []

# with torch.no_grad():
#     for mri, pet, y, apoe, age in full_loader:
#         mri = mri.to(device)
#         pet = pet.to(device)
#         y = y.to(device)

#         out = model(mri, pet, y)

#         all_z.append(out["z_sub"].cpu().numpy())
#         all_y.append(y.cpu().numpy())
#         all_apoe.append(apoe.numpy())
#         all_age.append(age.numpy())

# all_z = np.concatenate(all_z, axis=0)
# all_y = np.concatenate(all_y, axis=0)
# all_apoe = np.concatenate(all_apoe, axis=0)
# all_age = np.concatenate(all_age, axis=0)

# print("z_sub shape:", all_z.shape)


# # ------------------------------------------------------------
# # KMeans model selection
# # ------------------------------------------------------------
# n_samples = all_z.shape[0]
# max_k = min(10, n_samples - 1)
# Ks = list(range(2, max_k + 1))

# sil_scores = []
# inertias = []

# best_k = None
# best_ss = -1
# best_labels = None

# for K in Ks:
#     kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(all_z)

#     ss = silhouette_score(all_z, labels)

#     sil_scores.append(ss)
#     inertias.append(kmeans.inertia_)

#     print(f"K={K} | SS={ss:.4f} | Inertia={kmeans.inertia_:.2f}")

#     if ss > best_ss:
#         best_ss = ss
#         best_k = K
#         best_labels = labels

# print("\n======================")
# print(f"Best K = {best_k}")
# print(f"Best SS = {best_ss:.4f}")
# print("======================\n")


# # ------------------------------------------------------------
# # Save K-selection plots
# # ------------------------------------------------------------
# plt.figure()
# plt.plot(Ks, sil_scores, marker="o")
# plt.xlabel("K")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette vs K")
# plt.grid()
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "silhouette_vs_K.png"), dpi=150)
# plt.close()

# plt.figure()
# plt.plot(Ks, inertias, marker="o")
# plt.xlabel("K")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.grid()
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "elbow_vs_K.png"), dpi=150)
# plt.close()


# # ------------------------------------------------------------
# # Final clustering
# # ------------------------------------------------------------
# final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
# subtypes = final_kmeans.fit_predict(all_z)

# final_ss = silhouette_score(all_z, subtypes)


# # ------------------------------------------------------------
# # t-SNE visualization
# # ------------------------------------------------------------
# perplexity = min(30, max(5, (n_samples - 1) // 3))

# tsne = TSNE(
#     n_components=2,
#     random_state=42,
#     perplexity=perplexity,
#     init="pca",
#     learning_rate="auto",
# )

# z_2d = tsne.fit_transform(all_z)

# plt.figure()
# plt.scatter(z_2d[:, 0], z_2d[:, 1], c=subtypes, cmap="tab10", s=18)
# plt.title(f"Subtype t-SNE (K={best_k})")
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "tsne_bestK.png"), dpi=150)
# plt.close()


# # ------------------------------------------------------------
# # Save subtype assignments
# # ------------------------------------------------------------
# result_df = pd.DataFrame({
#     "PTID": full_mri["PTID"].astype(str).values,
#     "subtype": subtypes,
#     "DX_encoded": all_y,
#     "DX_label": label_encoder.inverse_transform(all_y),
#     "APOE4": all_apoe,
#     "AGE_z": all_age,
#     "AGE_recovered": all_age * age_std + age_mean,
# })

# result_df.to_csv(
#     os.path.join(save_dir, "subtype_with_bestK.csv"),
#     index=False,
# )

# def run_kmeans_ad_only(
#     all_z,
#     result_df,
#     save_dir,
#     dx_col="DX_label",
#     ad_label="AD",
#     k_min=2,
#     k_max=8,
#     random_state=42,
# ):
#     """
#     Run KMeans only on subjects with DX_label == AD.

#     Inputs:
#         all_z: np.ndarray of shape [N, z_dim]
#         result_df: dataframe with PTID and DX_label columns
#         save_dir: output directory
#         dx_col: diagnosis column in result_df
#         ad_label: label used for AD subjects
#     """

#     ad_save_dir = os.path.join(save_dir, "AD_only")
#     os.makedirs(ad_save_dir, exist_ok=True)

#     ad_mask = result_df[dx_col].astype(str).values == ad_label

#     z_ad = all_z[ad_mask]
#     result_ad = result_df.loc[ad_mask].copy().reset_index(drop=True)

#     print("\n======================")
#     print("AD-only clustering")
#     print(f"AD subjects: {len(result_ad)}")
#     print("======================\n")

#     if len(result_ad) < k_min + 1:
#         raise ValueError(f"Too few AD subjects for clustering: n={len(result_ad)}")

#     max_k = min(k_max, len(result_ad) - 1)
#     Ks = list(range(k_min, max_k + 1))

#     sil_scores = []
#     inertias = []

#     best_k = None
#     best_ss = -1
#     best_labels = None

#     for K in Ks:
#         kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
#         labels = kmeans.fit_predict(z_ad)

#         ss = silhouette_score(z_ad, labels)

#         sil_scores.append(ss)
#         inertias.append(kmeans.inertia_)

#         print(f"AD-only K={K} | SS={ss:.4f} | Inertia={kmeans.inertia_:.2f}")

#         if ss > best_ss:
#             best_ss = ss
#             best_k = K
#             best_labels = labels

#     print("\n======================")
#     print(f"Best AD-only K = {best_k}")
#     print(f"Best AD-only silhouette = {best_ss:.4f}")
#     print("======================\n")

#     # Save K-selection plots
#     plt.figure()
#     plt.plot(Ks, sil_scores, marker="o")
#     plt.xlabel("K")
#     plt.ylabel("Silhouette Score")
#     plt.title("AD-only Silhouette vs K")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(os.path.join(ad_save_dir, "AD_only_silhouette_vs_K.png"), dpi=150)
#     plt.close()

#     plt.figure()
#     plt.plot(Ks, inertias, marker="o")
#     plt.xlabel("K")
#     plt.ylabel("Inertia")
#     plt.title("AD-only Elbow Method")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(os.path.join(ad_save_dir, "AD_only_elbow_vs_K.png"), dpi=150)
#     plt.close()

#     # Final clustering
#     final_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
#     ad_subtypes = final_kmeans.fit_predict(z_ad)

#     final_ss = silhouette_score(z_ad, ad_subtypes)

#     result_ad["AD_subtype"] = ad_subtypes

#     # t-SNE only within AD
#     n_ad = z_ad.shape[0]
#     perplexity = min(30, max(5, (n_ad - 1) // 3))

#     tsne = TSNE(
#         n_components=2,
#         random_state=random_state,
#         perplexity=perplexity,
#         init="pca",
#         learning_rate="auto",
#     )

#     z_ad_2d = tsne.fit_transform(z_ad)

# # ------
#     # Store t-SNE coordinates
#     result_ad["tsne_1"] = z_ad_2d[:, 0]
#     result_ad["tsne_2"] = z_ad_2d[:, 1]

#     # ------------------------------------------------------------
#     # Plot 1: AD-only t-SNE colored by AD subtype
#     # ------------------------------------------------------------
#     plt.figure(figsize=(7, 5))
#     plt.scatter(
#         z_ad_2d[:, 0],
#         z_ad_2d[:, 1],
#         c=ad_subtypes,
#         cmap="tab10",
#         s=22,
#         alpha=0.85,
#     )
#     plt.title(f"AD-only subtype t-SNE (K={best_k})")
#     plt.xlabel("t-SNE 1")
#     plt.ylabel("t-SNE 2")
#     plt.tight_layout()
#     plt.savefig(os.path.join(ad_save_dir, "AD_only_tsne_bestK.png"), dpi=150)
#     plt.close()

#     # ------------------------------------------------------------
#     # Plot 2: AD-only t-SNE colored by APOE4
#     # ------------------------------------------------------------
#     plt.figure(figsize=(7, 5))
#     scatter = plt.scatter(
#         z_ad_2d[:, 0],
#         z_ad_2d[:, 1],
#         c=result_ad["APOE4"].astype(int),
#         cmap="viridis",
#         s=22,
#         alpha=0.85,
#     )
#     cbar = plt.colorbar(scatter)
#     cbar.set_label("APOE4 count")
#     cbar.set_ticks([0, 1, 2])
#     plt.title("AD-only t-SNE colored by APOE4")
#     plt.xlabel("t-SNE 1")
#     plt.ylabel("t-SNE 2")
#     plt.tight_layout()
#     plt.savefig(os.path.join(ad_save_dir, "AD_only_tsne_by_APOE4.png"), dpi=150)
#     plt.close()

#     # ------------------------------------------------------------
#     # Plot 3: AD-only t-SNE colored by age
#     # ------------------------------------------------------------
#     plt.figure(figsize=(7, 5))
#     scatter = plt.scatter(
#         z_ad_2d[:, 0],
#         z_ad_2d[:, 1],
#         c=result_ad["AGE_recovered"],
#         cmap="viridis",
#         s=22,
#         alpha=0.85,
#     )
#     cbar = plt.colorbar(scatter)
#     cbar.set_label("Age")
#     plt.title("AD-only t-SNE colored by age")
#     plt.xlabel("t-SNE 1")
#     plt.ylabel("t-SNE 2")
#     plt.tight_layout()
#     plt.savefig(os.path.join(ad_save_dir, "AD_only_tsne_by_AGE.png"), dpi=150)
#     plt.close()


#     # plt.figure()
#     # plt.scatter(z_ad_2d[:, 0], z_ad_2d[:, 1], c=ad_subtypes, cmap="tab10", s=18)
#     # plt.title(f"AD-only subtype t-SNE (K={best_k})")
#     # plt.tight_layout()
#     # plt.savefig(os.path.join(ad_save_dir, "AD_only_tsne_bestK.png"), dpi=150)
#     # plt.close()

#     # ------------------------------------------------------------
#     # AD-only subtype summaries
#     # ------------------------------------------------------------
#     ad_summary = result_ad.groupby("AD_subtype")[["AGE_recovered", "APOE4"]].agg(
#         {
#             "AGE_recovered": ["mean", "std", "median", "count"],
#             "APOE4": ["mean", "std", "median"],
#         }
#     )

#     ad_summary.to_csv(os.path.join(ad_save_dir, "AD_only_subtype_summary.csv"))

#     apoe_pct = pd.crosstab(
#         result_ad["AD_subtype"],
#         result_ad["APOE4"],
#         normalize="index",
#     ) * 100

#     apoe_cnt = pd.crosstab(
#         result_ad["AD_subtype"],
#         result_ad["APOE4"],
#     )

#     apoe_pct.to_csv(os.path.join(ad_save_dir, "AD_only_APOE4_percent.csv"))
#     apoe_cnt.to_csv(os.path.join(ad_save_dir, "AD_only_APOE4_count.csv"))

#     print("\n=== AD-only subtype summary ===")
#     print(ad_summary)

#     print("\n=== AD-only APOE4 percent ===")
#     print(apoe_pct)


#     # Save AD-only subtype assignments
#     result_ad.to_csv(
#         os.path.join(ad_save_dir, "AD_only_subtype_with_bestK.csv"),
#         index=False,
#     )

#     print("\n======================")
#     print(f"Final AD-only K: {best_k}")
#     print(f"Final AD-only silhouette: {final_ss:.4f}")
#     print("======================\n")

#     print("Saved AD-only results to:", ad_save_dir)

#     return result_ad, best_k, final_ss

# result_ad, best_k_ad, final_ss_ad = run_kmeans_ad_only(
#     all_z=all_z,
#     result_df=result_df,
#     save_dir=save_dir,
#     dx_col="DX_label",
#     ad_label="AD",
#     k_min=2,
#     k_max=8,
# )

# # ------------------------------------------------------------
# # Compare subtypes against DX_bl / diagnosis
# # ------------------------------------------------------------
# char_df = pd.read_csv("./dataset/characteristic.csv")
# char_df["PTID"] = char_df["PTID"].astype(str)

# merged = pd.merge(result_df, char_df, on="PTID", how="inner")
# merged = merged.dropna(subset=["DX_bl"])

# merged["DX_label_factorized"] = pd.factorize(merged["DX_bl"])[0]

# ari = adjusted_rand_score(
#     merged["DX_label_factorized"],
#     merged["subtype"],
# )

# dx_count = pd.crosstab(merged["subtype"], merged["DX_bl"])
# dx_pct = pd.crosstab(merged["subtype"], merged["DX_bl"], normalize="index") * 100

# dx_count.to_csv(os.path.join(save_dir, "subtype_dx_count.csv"))
# dx_pct.to_csv(os.path.join(save_dir, "subtype_dx_percent.csv"))

# print("\n=== Subtype vs DX_bl count ===")
# print(dx_count)

# print("\n=== Subtype vs DX_bl percent ===")
# print(dx_pct)

# print("\n======================")
# print(f"Final K: {best_k}")
# print(f"Silhouette Score: {final_ss:.4f}")
# print(f"ARI (vs DX_bl): {ari:.4f}")
# print("======================\n")

# print("Saved to:", save_dir)