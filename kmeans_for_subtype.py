import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from model import AE
from dataset import ROIDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./results/best.pth"
save_dir = "./results/subtype_analysis"
os.makedirs(save_dir, exist_ok=True)


mri_df = pd.read_csv("./dataset/MRI_CN_AD.csv")
pet_df = pd.read_csv("./dataset/PET_CN_AD.csv")

assert "PTID" in mri_df.columns
assert "PTID" in pet_df.columns

full_mri = mri_df.reset_index(drop=True)
full_pet = pet_df.reset_index(drop=True)

orig_dataset = ROIDataset(
    full_mri,
    full_pet,
    fit=True
)

mri_scaler = orig_dataset.mri_scaler
pet_scaler = orig_dataset.pet_scaler

# 再构建 dataset（不fit）
full_dataset = ROIDataset(
    full_mri,
    full_pet,
    mri_scaler=mri_scaler,
    pet_scaler=pet_scaler,
    label_encoder=orig_dataset.label_encoder,
    fit=False
)

full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

model = AE(num_rois=162, latent=128, dim=512, beta_mi=0.1, beta_ib=0.1, dropout_p=0.0, num_classes=len(train_dataset.label_encoder.classes_))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded.")

all_z = []

with torch.no_grad():
    for mri, pet, y in full_loader:
        mri = mri.to(device)
        pet = pet.to(device)
        y = y.to(device)

        out = model(mri, pet, y)
        all_z.append(out["z_sub"].cpu().numpy())

all_z = np.concatenate(all_z, axis=0)

print("z_sub shape:", all_z.shape)

Ks = list(range(2, 11))

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

plt.figure()
plt.plot(Ks, sil_scores, marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette vs K")
plt.grid()
plt.savefig(os.path.join(save_dir, "silhouette_vs_K.png"), dpi=150)
plt.close()

plt.figure()
plt.plot(Ks, inertias, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()
plt.savefig(os.path.join(save_dir, "elbow_vs_K.png"), dpi=150)
plt.close()

final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
subtypes = final_kmeans.fit_predict(all_z)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_2d = tsne.fit_transform(all_z)

plt.figure()
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=subtypes, cmap="tab10")
plt.title(f"Subtype t-SNE (K={best_k})")
plt.savefig(os.path.join(save_dir, "tsne_bestK.png"), dpi=150)
plt.close()

final_ss = silhouette_score(all_z, subtypes)

char_df = pd.read_csv("./dataset/characteristic.csv")
char_df["PTID"] = char_df["PTID"].astype(str)

result_df = pd.DataFrame({
    "PTID": test_mri["PTID"].values,
    "subtype": subtypes
})

merged = pd.merge(result_df, char_df, on="PTID", how="inner")
merged = merged.dropna(subset=["DX_bl"])

merged["DX_label"] = pd.factorize(merged["DX_bl"])[0]

ari = adjusted_rand_score(merged["DX_label"], merged["subtype"])

result_df.to_csv(
    os.path.join(save_dir, "subtype_with_bestK.csv"),
    index=False
)

print("\n======================")
print(f"Final K: {best_k}")
print(f"Silhouette Score: {final_ss:.4f}")
print(f"ARI (vs DX_bl): {ari:.4f}")
print("======================\n")

print("Saved to:", save_dir)