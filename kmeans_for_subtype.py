import os
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

# ======================
# setup
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./results/best.pth"
save_dir = "./results/subtype_analysis"
os.makedirs(save_dir, exist_ok=True)

# ======================
# load data
# ======================
train_mri = pd.read_csv("./train_mri.csv")
train_pet = pd.read_csv("./train_pet.csv")

val_mri = pd.read_csv("./val_mri.csv")
val_pet = pd.read_csv("./val_pet.csv")

test_mri = pd.read_csv("./test_mri.csv")
test_pet = pd.read_csv("./test_pet.csv")

# ======================
# align
# ======================
train_mri = train_mri.sort_values("PTID").reset_index(drop=True)
train_pet = train_pet.sort_values("PTID").reset_index(drop=True)

val_mri = val_mri.sort_values("PTID").reset_index(drop=True)
val_pet = val_pet.sort_values("PTID").reset_index(drop=True)

test_mri = test_mri.sort_values("PTID").reset_index(drop=True)
test_pet = test_pet.sort_values("PTID").reset_index(drop=True)

assert all(train_mri["PTID"] == train_pet["PTID"])
assert all(val_mri["PTID"] == val_pet["PTID"])
assert all(test_mri["PTID"] == test_pet["PTID"])

# ======================
# dataset + scaler
# ======================
train_dataset = ROIDataset(train_mri, train_pet, fit=True)

mri_scaler = train_dataset.mri_scaler
pet_scaler = train_dataset.pet_scaler
label_encoder = train_dataset.label_encoder

val_dataset = ROIDataset(
    val_mri, val_pet,
    mri_scaler=mri_scaler,
    pet_scaler=pet_scaler,
    label_encoder=label_encoder,
    gender_encoder=train_dataset.gender_encoder,
    fit=False
)

test_dataset = ROIDataset(
    test_mri, test_pet,
    mri_scaler=mri_scaler,
    pet_scaler=pet_scaler,
    label_encoder=label_encoder,
    gender_encoder=train_dataset.gender_encoder,
    fit=False
)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ======================
# model
# ======================
model = AE(
    num_rois=162,
    latent=128,
    dim=512,
    beta_mi=0.1,
    beta_ib=0.1,
    l1_lambda=0.0001,
    beta_ib_gender = 0.1,
    dropout_p=0.0,
    num_classes=len(train_dataset.label_encoder.classes_)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded.")

# ======================
# extract VAL embeddings
# ======================
val_z = []

with torch.no_grad():
    for mri, pet, y, y_gender in val_loader:
        mri = mri.to(device)
        pet = pet.to(device)
        y = y.to(device)
        y_gender = y_gender.to(device)

        out = model(mri, pet, y, y_gender)
        val_z.append(out["z_sub"].cpu().numpy())

val_z = np.concatenate(val_z, axis=0)

print("val_z shape:", val_z.shape)

# ======================
# K selection on VAL
# ======================
Ks = list(range(2, 11))

sil_scores = []
inertias = []
best_k = None
best_ss = -1

for K in Ks:
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(val_z)

    ss = silhouette_score(val_z, labels)

    sil_scores.append(ss)
    inertias.append(kmeans.inertia_)

    print(f"K={K} | SS={ss:.4f} | Inertia={kmeans.inertia_:.2f}")

    if ss > best_ss:
        best_ss = ss
        best_k = K

print("\n======================")
print(f"Best K = {best_k}")
print(f"Best SS = {best_ss:.4f}")
print("======================\n")

# ======================
# plots (VAL)
# ======================
plt.figure()
plt.plot(Ks, sil_scores, marker="o")
plt.title("Silhouette vs K (VAL)")
plt.grid()
plt.savefig(os.path.join(save_dir, "silhouette_val.png"), dpi=150)
plt.close()

plt.figure()
plt.plot(Ks, inertias, marker="o")
plt.title("Elbow (VAL)")
plt.grid()
plt.savefig(os.path.join(save_dir, "elbow_val.png"), dpi=150)
plt.close()

# ======================
# FINAL KMeans (fit on VAL)
# ======================
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_kmeans.fit(val_z)

val_subtypes = final_kmeans.predict(val_z)

# ======================
# TEST embeddings
# ======================
test_z = []

with torch.no_grad():
    for mri, pet, y, y_gender in test_loader:
        mri = mri.to(device)
        pet = pet.to(device)
        y = y.to(device)
        y_gender = y_gender.to(device)

        out = model(mri, pet, y, y_gender)
        test_z.append(out["z_sub"].cpu().numpy())

test_z = np.concatenate(test_z, axis=0)

print("test_z shape:", test_z.shape)

# ======================
# TEST prediction (NO FIT)
# ======================
test_subtypes = final_kmeans.predict(test_z)

# ======================
# TEST t-SNE
# ======================
perplexity = min(30, len(test_z) - 1)

tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
test_z_2d = tsne.fit_transform(test_z)

plt.figure()
plt.scatter(test_z_2d[:, 0], test_z_2d[:, 1], c=test_subtypes, cmap="tab10")
plt.title(f"TEST t-SNE (K={best_k})")
plt.savefig(os.path.join(save_dir, "tsne_test.png"), dpi=150)
plt.close()

# ======================
# VAL t-SNE (optional)
# ======================
val_z_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(val_z)-1)).fit_transform(val_z)

plt.figure()
plt.scatter(val_z_2d[:, 0], val_z_2d[:, 1], c=val_subtypes, cmap="tab10")
plt.title(f"VAL t-SNE (K={best_k})")
plt.savefig(os.path.join(save_dir, "tsne_val.png"), dpi=150)
plt.close()

# ======================
# ARI (TEST only)
# ======================
char_df = pd.read_csv("./dataset/characteristic.csv")
char_df["PTID"] = char_df["PTID"].astype(str)

result_df = pd.DataFrame({
    "PTID": test_mri["PTID"].astype(str).values,
    "subtype": test_subtypes
})

merged = pd.merge(result_df, char_df, on="PTID", how="inner")
merged = merged.dropna(subset=["DX_bl"])
merged["DX_label"] = pd.factorize(merged["DX_bl"])[0]

ari = adjusted_rand_score(merged["DX_label"], merged["subtype"])

result_df.to_csv(
    os.path.join(save_dir, "test_subtype_with_bestK.csv"),
    index=False
)

# ======================
# final metrics
# ======================
final_ss = silhouette_score(test_z, test_subtypes)

print("\n======================")
print(f"Final K: {best_k}")
print(f"TEST Silhouette: {final_ss:.4f}")
print(f"ARI (vs DX_bl): {ari:.4f}")
print("======================\n")

print("Saved to:", save_dir)