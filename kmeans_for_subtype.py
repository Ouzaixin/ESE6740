import os
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE

from model import AE
from dataset import ROIDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./results/best.pth"
preprocess_path = "./results/preprocess_objects.pkl"
save_dir = "./results/subtype_analysis"
os.makedirs(save_dir, exist_ok=True)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
mri_df = pd.read_csv("./dataset/MRI_CN_AD.csv")
pet_df = pd.read_csv("./dataset/PET_CN_AD.csv")

assert "PTID" in mri_df.columns
assert "PTID" in pet_df.columns

full_mri = mri_df.reset_index(drop=True)
full_pet = pet_df.reset_index(drop=True)


# ------------------------------------------------------------
# Load train-time preprocessing objects
# ------------------------------------------------------------
with open(preprocess_path, "rb") as f:
    preprocess_objects = pickle.load(f)

mri_imputer = preprocess_objects["mri_imputer"]
pet_imputer = preprocess_objects["pet_imputer"]
mri_scaler = preprocess_objects["mri_scaler"]
pet_scaler = preprocess_objects["pet_scaler"]
label_encoder = preprocess_objects["label_encoder"]
age_mean = preprocess_objects["age_mean"]
age_std = preprocess_objects["age_std"]
apoe_fill = preprocess_objects.get("apoe_fill", 0)


# ------------------------------------------------------------
# Build full dataset using train-time preprocessing
# ------------------------------------------------------------
full_dataset = ROIDataset(
    full_mri,
    full_pet,
    mri_imputer=mri_imputer,
    pet_imputer=pet_imputer,
    mri_scaler=mri_scaler,
    pet_scaler=pet_scaler,
    label_encoder=label_encoder,
    age_mean=age_mean,
    age_std=age_std,
    apoe_fill=apoe_fill,
    fit=False,
)

full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)


# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------
num_rois = full_dataset.X_mri.shape[1]
latent = 128
# subtype_dim = latent // 32
subtype_dim = latent // 16

model = AE(
    num_rois=num_rois,
    latent=latent,
    dim=512,
    beta_mi=1.0,
    beta_ib=2.0,
    subtype_dim=subtype_dim,
    dropout_p=0.0,
    num_classes=len(label_encoder.classes_),
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded.")
print(f"num_rois = {num_rois}")
print(f"subtype_dim = {subtype_dim}")
print(f"classes = {list(label_encoder.classes_)}")

# ------------------------------------------------------------
# Extract z_sub for all subjects
# ------------------------------------------------------------
all_z = []
all_y = []
all_apoe = []
all_age = []

with torch.no_grad():
    for mri, pet, y, apoe, age in full_loader:
        mri = mri.to(device)
        pet = pet.to(device)
        y = y.to(device)

        out = model(mri, pet, y)

        all_z.append(out["z_sub"].cpu().numpy())
        all_y.append(y.cpu().numpy())
        all_apoe.append(apoe.numpy())
        all_age.append(age.numpy())

all_z = np.concatenate(all_z, axis=0)
all_y = np.concatenate(all_y, axis=0)
all_apoe = np.concatenate(all_apoe, axis=0)
all_age = np.concatenate(all_age, axis=0)

print("z_sub shape:", all_z.shape)


# ------------------------------------------------------------
# KMeans model selection
# ------------------------------------------------------------
n_samples = all_z.shape[0]
max_k = min(10, n_samples - 1)
Ks = list(range(2, max_k + 1))

sil_scores = []
inertias = []

best_k = None
best_ss = -1
best_labels = None

for K in Ks:
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_z)

    ss = silhouette_score(all_z, labels)

    sil_scores.append(ss)
    inertias.append(kmeans.inertia_)

    print(f"K={K} | SS={ss:.4f} | Inertia={kmeans.inertia_:.2f}")

    if ss > best_ss:
        best_ss = ss
        best_k = K
        best_labels = labels

print("\n======================")
print(f"Best K = {best_k}")
print(f"Best SS = {best_ss:.4f}")
print("======================\n")


# ------------------------------------------------------------
# Save K-selection plots
# ------------------------------------------------------------
plt.figure()
plt.plot(Ks, sil_scores, marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette vs K")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "silhouette_vs_K.png"), dpi=150)
plt.close()

plt.figure()
plt.plot(Ks, inertias, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "elbow_vs_K.png"), dpi=150)
plt.close()


# ------------------------------------------------------------
# Final clustering
# ------------------------------------------------------------
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
subtypes = final_kmeans.fit_predict(all_z)

final_ss = silhouette_score(all_z, subtypes)


# ------------------------------------------------------------
# t-SNE visualization
# ------------------------------------------------------------
perplexity = min(30, max(5, (n_samples - 1) // 3))

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=perplexity,
    init="pca",
    learning_rate="auto",
)

z_2d = tsne.fit_transform(all_z)

plt.figure()
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=subtypes, cmap="tab10", s=18)
plt.title(f"Subtype t-SNE (K={best_k})")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "tsne_bestK.png"), dpi=150)
plt.close()


# ------------------------------------------------------------
# Save subtype assignments
# ------------------------------------------------------------
result_df = pd.DataFrame({
    "PTID": full_mri["PTID"].astype(str).values,
    "subtype": subtypes,
    "DX_encoded": all_y,
    "DX_label": label_encoder.inverse_transform(all_y),
    "APOE4": all_apoe,
    "AGE_z": all_age,
    "AGE_recovered": all_age * age_std + age_mean,
})

result_df.to_csv(
    os.path.join(save_dir, "subtype_with_bestK.csv"),
    index=False,
)

def run_kmeans_ad_only(
    all_z,
    result_df,
    save_dir,
    dx_col="DX_label",
    ad_label="AD",
    k_min=2,
    k_max=8,
    random_state=42,
):
    """
    Run KMeans only on subjects with DX_label == AD.

    Inputs:
        all_z: np.ndarray of shape [N, z_dim]
        result_df: dataframe with PTID and DX_label columns
        save_dir: output directory
        dx_col: diagnosis column in result_df
        ad_label: label used for AD subjects
    """

    ad_save_dir = os.path.join(save_dir, "AD_only")
    os.makedirs(ad_save_dir, exist_ok=True)

    ad_mask = result_df[dx_col].astype(str).values == ad_label

    z_ad = all_z[ad_mask]
    result_ad = result_df.loc[ad_mask].copy().reset_index(drop=True)

    print("\n======================")
    print("AD-only clustering")
    print(f"AD subjects: {len(result_ad)}")
    print("======================\n")

    if len(result_ad) < k_min + 1:
        raise ValueError(f"Too few AD subjects for clustering: n={len(result_ad)}")

    max_k = min(k_max, len(result_ad) - 1)
    Ks = list(range(k_min, max_k + 1))

    sil_scores = []
    inertias = []

    best_k = None
    best_ss = -1
    best_labels = None

    for K in Ks:
        kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(z_ad)

        ss = silhouette_score(z_ad, labels)

        sil_scores.append(ss)
        inertias.append(kmeans.inertia_)

        print(f"AD-only K={K} | SS={ss:.4f} | Inertia={kmeans.inertia_:.2f}")

        if ss > best_ss:
            best_ss = ss
            best_k = K
            best_labels = labels

    print("\n======================")
    print(f"Best AD-only K = {best_k}")
    print(f"Best AD-only silhouette = {best_ss:.4f}")
    print("======================\n")

    # Save K-selection plots
    plt.figure()
    plt.plot(Ks, sil_scores, marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.title("AD-only Silhouette vs K")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(ad_save_dir, "AD_only_silhouette_vs_K.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(Ks, inertias, marker="o")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("AD-only Elbow Method")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(ad_save_dir, "AD_only_elbow_vs_K.png"), dpi=150)
    plt.close()

    # Final clustering
    final_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    ad_subtypes = final_kmeans.fit_predict(z_ad)

    final_ss = silhouette_score(z_ad, ad_subtypes)

    result_ad["AD_subtype"] = ad_subtypes

    # t-SNE only within AD
    n_ad = z_ad.shape[0]
    perplexity = min(30, max(5, (n_ad - 1) // 3))

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )

    z_ad_2d = tsne.fit_transform(z_ad)

# ------
    # Store t-SNE coordinates
    result_ad["tsne_1"] = z_ad_2d[:, 0]
    result_ad["tsne_2"] = z_ad_2d[:, 1]

    # ------------------------------------------------------------
    # Plot 1: AD-only t-SNE colored by AD subtype
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.scatter(
        z_ad_2d[:, 0],
        z_ad_2d[:, 1],
        c=ad_subtypes,
        cmap="tab10",
        s=22,
        alpha=0.85,
    )
    plt.title(f"AD-only subtype t-SNE (K={best_k})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(ad_save_dir, "AD_only_tsne_bestK.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: AD-only t-SNE colored by APOE4
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        z_ad_2d[:, 0],
        z_ad_2d[:, 1],
        c=result_ad["APOE4"].astype(int),
        cmap="viridis",
        s=22,
        alpha=0.85,
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("APOE4 count")
    cbar.set_ticks([0, 1, 2])
    plt.title("AD-only t-SNE colored by APOE4")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(ad_save_dir, "AD_only_tsne_by_APOE4.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: AD-only t-SNE colored by age
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        z_ad_2d[:, 0],
        z_ad_2d[:, 1],
        c=result_ad["AGE_recovered"],
        cmap="viridis",
        s=22,
        alpha=0.85,
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Age")
    plt.title("AD-only t-SNE colored by age")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(ad_save_dir, "AD_only_tsne_by_AGE.png"), dpi=150)
    plt.close()


    # plt.figure()
    # plt.scatter(z_ad_2d[:, 0], z_ad_2d[:, 1], c=ad_subtypes, cmap="tab10", s=18)
    # plt.title(f"AD-only subtype t-SNE (K={best_k})")
    # plt.tight_layout()
    # plt.savefig(os.path.join(ad_save_dir, "AD_only_tsne_bestK.png"), dpi=150)
    # plt.close()

    # ------------------------------------------------------------
    # AD-only subtype summaries
    # ------------------------------------------------------------
    ad_summary = result_ad.groupby("AD_subtype")[["AGE_recovered", "APOE4"]].agg(
        {
            "AGE_recovered": ["mean", "std", "median", "count"],
            "APOE4": ["mean", "std", "median"],
        }
    )

    ad_summary.to_csv(os.path.join(ad_save_dir, "AD_only_subtype_summary.csv"))

    apoe_pct = pd.crosstab(
        result_ad["AD_subtype"],
        result_ad["APOE4"],
        normalize="index",
    ) * 100

    apoe_cnt = pd.crosstab(
        result_ad["AD_subtype"],
        result_ad["APOE4"],
    )

    apoe_pct.to_csv(os.path.join(ad_save_dir, "AD_only_APOE4_percent.csv"))
    apoe_cnt.to_csv(os.path.join(ad_save_dir, "AD_only_APOE4_count.csv"))

    print("\n=== AD-only subtype summary ===")
    print(ad_summary)

    print("\n=== AD-only APOE4 percent ===")
    print(apoe_pct)


    # Save AD-only subtype assignments
    result_ad.to_csv(
        os.path.join(ad_save_dir, "AD_only_subtype_with_bestK.csv"),
        index=False,
    )

    print("\n======================")
    print(f"Final AD-only K: {best_k}")
    print(f"Final AD-only silhouette: {final_ss:.4f}")
    print("======================\n")

    print("Saved AD-only results to:", ad_save_dir)

    return result_ad, best_k, final_ss

result_ad, best_k_ad, final_ss_ad = run_kmeans_ad_only(
    all_z=all_z,
    result_df=result_df,
    save_dir=save_dir,
    dx_col="DX_label",
    ad_label="AD",
    k_min=2,
    k_max=8,
)

# ------------------------------------------------------------
# Compare subtypes against DX_bl / diagnosis
# ------------------------------------------------------------
char_df = pd.read_csv("./dataset/characteristic.csv")
char_df["PTID"] = char_df["PTID"].astype(str)

merged = pd.merge(result_df, char_df, on="PTID", how="inner")
merged = merged.dropna(subset=["DX_bl"])

merged["DX_label_factorized"] = pd.factorize(merged["DX_bl"])[0]

ari = adjusted_rand_score(
    merged["DX_label_factorized"],
    merged["subtype"],
)

dx_count = pd.crosstab(merged["subtype"], merged["DX_bl"])
dx_pct = pd.crosstab(merged["subtype"], merged["DX_bl"], normalize="index") * 100

dx_count.to_csv(os.path.join(save_dir, "subtype_dx_count.csv"))
dx_pct.to_csv(os.path.join(save_dir, "subtype_dx_percent.csv"))

print("\n=== Subtype vs DX_bl count ===")
print(dx_count)

print("\n=== Subtype vs DX_bl percent ===")
print(dx_pct)

print("\n======================")
print(f"Final K: {best_k}")
print(f"Silhouette Score: {final_ss:.4f}")
print(f"ARI (vs DX_bl): {ari:.4f}")
print("======================\n")

print("Saved to:", save_dir)