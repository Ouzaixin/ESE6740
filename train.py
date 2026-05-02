import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

from model import AE, CLUB
from dataset import ROIDataset
from utils import visualize

matplotlib.use("Agg")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, club, train_dataset, val_dataset, device, epochs=100, lr=1e-4, batch_size=64, plot_path="loss_curve.png"):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    club.to(device)
    opt_club = torch.optim.Adam(club.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model = None

    train_losses = []
    rec_losses = []
    cls_losses = []
    val_losses = []
    mi_losses = []
    mi_estimates = []

    plt.figure()

    for ep in range(epochs):

        total_train_loss = 0
        total_rec_loss = 0
        total_cls_loss = 0
        total_mi_loss = 0
        mi_estimate = 0

        for mri, pet, y in train_loader:
            model.train()
            club.eval()
            mri = mri.to(device)
            pet = pet.to(device)
            y = y.to(device)

            out = model(mri, pet, y)
            mi = club(out["z_mod"], out["z_sub"])
            loss = out["loss"] + model.beta_mi * mi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for _ in range(10):
                club.train()
                mi_loss = club.learning_loss(out["z_mod"].detach(), out["z_sub"].detach())
                opt_club.zero_grad()
                mi_loss.backward()
                opt_club.step()

            total_train_loss += out["loss"].item() * mri.size(0)
            total_rec_loss += out["loss_rec"].item() * mri.size(0)
            total_cls_loss += out["loss_cls"].item() * mri.size(0)
            
            total_mi_loss += mi_loss.item() * mri.size(0)
            mi_estimate += mi.item() * mri.size(0)

        total_train_loss /= len(train_dataset)
        total_rec_loss /= len(train_dataset)
        total_cls_loss /= len(train_dataset)

        mi_estimate /= len(train_dataset)
        total_mi_loss /= len(train_dataset)

        train_losses.append(total_train_loss)
        rec_losses.append(total_rec_loss)
        cls_losses.append(total_cls_loss)
        mi_losses.append(total_mi_loss)
        mi_estimates.append(mi_estimate)

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for mri, pet, y in val_loader:
                mri = mri.to(device)
                pet = pet.to(device)
                y = y.to(device)
                out = model(mri, pet, y)
                loss = out["loss"]
                val_loss += loss.item() * mri.size(0)

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

            # visualize(
            #     mri[0],
            #     pet[0],
            #     out["reconstruction_m"][0],
            #     out["reconstruction_p"][0],
            #     save_path=os.path.join(dir, f"val_epoch_{ep+1}.png")
            # )

        plt.clf()
        plt.plot(train_losses, label="train")
        plt.plot(rec_losses, label="Reconstruction Loss")
        plt.plot(cls_losses, label="Classification Loss")
        plt.plot(val_losses, label="val")
        plt.plot(mi_losses, label="MI_loss")
        plt.plot(mi_estimates, label="MI Estimate")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Training Curve")

        os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Epoch {ep+1}/{epochs} | Train {total_train_loss:.4f} | Rec_Loss {total_rec_loss:.4f} | Cls_Loss {total_cls_loss:.4f} | Val {val_loss:.4f} | MI_Loss {total_mi_loss:.4f} | MI_Estimate {mi_estimate:.4f} |")

    return best_val_loss, best_model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    dir = "./results"
    os.makedirs(dir, exist_ok=True)

    mri_df = pd.read_csv("./dataset/MRI_CN_AD.csv")
    pet_df = pd.read_csv("./dataset/PET_CN_AD.csv")

    indices = mri_df.index.values

    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_mri = mri_df.iloc[train_idx].reset_index(drop=True)
    val_mri   = mri_df.iloc[val_idx].reset_index(drop=True)
    test_mri  = mri_df.iloc[test_idx].reset_index(drop=True)

    train_pet = pet_df.iloc[train_idx].reset_index(drop=True)
    val_pet   = pet_df.iloc[val_idx].reset_index(drop=True)
    test_pet  = pet_df.iloc[test_idx].reset_index(drop=True)

    train_dataset = ROIDataset(train_mri, train_pet, fit=True)

    # get scalers from train
    label_encoder = train_dataset.label_encoder
    mri_scaler = train_dataset.mri_scaler
    pet_scaler = train_dataset.pet_scaler

    val_dataset = ROIDataset(
        val_mri,
        val_pet,
        mri_scaler=mri_scaler,
        pet_scaler=pet_scaler,
        label_encoder=label_encoder,
        fit=False
    )

    test_dataset = ROIDataset(
        test_mri,
        test_pet,
        mri_scaler=mri_scaler,
        pet_scaler=pet_scaler,
        label_encoder=label_encoder,
        fit=False
    )

    # model training
    model = AE(num_rois=162, latent=128, dim=512, beta_mi=0.1, beta_ib=0.1, dropout_p=0.0, num_classes=len(label_encoder.classes_))
    club = CLUB(x_dim=128, y_dim=128//32, hidden_size=128).to(device)

    best_val_loss, best_model = train_model(model, club, train_dataset, val_dataset, device, epochs = 10000, lr = 1e-4, batch_size = 64, plot_path=os.path.join(dir, "loss_curve.png"))

    torch.save(best_model, f"{dir}/best.pth")
    print("Best Val Loss:", best_val_loss)

    # test evaluation
    model.load_state_dict(best_model)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        total_loss = 0
        total_n = 0
        for idx, (mri, pet, y) in enumerate(test_loader):
            mri = mri.to(device)
            pet = pet.to(device)
            y = y.to(device)

            out = model(mri, pet, y)

            if idx == 0:
                for i in range(min(5, mri.size(0))):
                    visualize(
                        mri[i],
                        pet[i],
                        out["reconstruction_m"][i],
                        out["reconstruction_p"][i],
                        save_path=os.path.join(dir, f"test_epoch_{i+1}.png")
                    )

            batch_size = mri.size(0)
            total_loss += out["loss"].item() * batch_size
            total_n += batch_size

        print("Test Loss:", total_loss / total_n)