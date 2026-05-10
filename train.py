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
    l1_losses = []
    cls_gender_losses = []
    val_losses = []
    mi_losses = []
    mi_estimates = []

    sub_pred_m_mean = []
    sub_pred_p_mean = []
    sub_pred_m_std = []
    sub_pred_p_std = []

    plt.figure()

    for ep in range(epochs):

        total_train_loss = 0
        total_rec_loss = 0
        total_cls_loss = 0
        total_l1_loss = 0
        total_cls_gender_loss = 0
        total_mi_loss = 0
        mi_estimate = 0

        for mri, pet, y, y_gender in train_loader:
            model.train()
            club.eval()
            mri = mri.to(device)
            pet = pet.to(device)
            y = y.to(device)
            y_gender = y_gender.to(device)

            out = model(mri, pet, y, y_gender)
            mi = club(out["z_mod"], out["z_sub"])
            loss = out["loss"] + model.beta_mi * mi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for _ in range(20):
                club.train()
                mi_loss = club.learning_loss(out["z_mod"].detach(), out["z_sub"].detach())
                opt_club.zero_grad()
                mi_loss.backward()
                opt_club.step()

            total_train_loss += out["loss"].item() * mri.size(0)
            total_rec_loss += out["loss_rec"].item() * mri.size(0)
            total_cls_loss += out["loss_cls"].item() * mri.size(0)
            total_l1_loss += out["loss_l1"].item() * mri.size(0)
            total_cls_gender_loss += out["loss_cls_gender"].item() * mri.size(0)
            
            total_mi_loss += mi_loss.item() * mri.size(0)
            mi_estimate += mi.item() * mri.size(0)

        sub_pred_m_mean.append(out["sub_pred_m"].mean().item())
        sub_pred_m_std.append(out["sub_pred_m"].std().item())

        sub_pred_p_mean.append(out["sub_pred_p"].mean().item())
        sub_pred_p_std.append(out["sub_pred_p"].std().item())

        total_train_loss /= len(train_dataset)
        total_rec_loss /= len(train_dataset)
        total_cls_loss /= len(train_dataset)
        total_l1_loss /= len(train_dataset)
        total_cls_gender_loss /= len(train_dataset)

        mi_estimate /= len(train_dataset)
        total_mi_loss /= len(train_dataset)

        train_losses.append(total_train_loss)
        rec_losses.append(total_rec_loss)
        cls_losses.append(total_cls_loss)
        l1_losses.append(total_l1_loss)
        cls_gender_losses.append(total_cls_gender_loss)

        mi_losses.append(total_mi_loss)
        mi_estimates.append(mi_estimate)

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for mri, pet, y, y_gender in val_loader:
                mri = mri.to(device)
                pet = pet.to(device)
                y = y.to(device)
                y_gender = y_gender.to(device)
                out = model(mri, pet, y, y_gender)
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

        fig, axes = plt.subplots(4, 3, figsize=(16, 14))

        # =========================
        # (1) Total train loss
        # =========================
        axes[0, 0].plot(train_losses, label="train total")
        axes[0, 0].plot(val_losses, label="val")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("epoch")
        axes[0, 0].set_ylabel("loss")
        axes[0, 0].legend()
        axes[0, 0].grid()

        # =========================
        # (2) Reconstruction loss
        # =========================
        axes[0, 1].plot(rec_losses, label="reconstruction")
        axes[0, 1].set_title("Reconstruction Loss")
        axes[0, 1].set_xlabel("epoch")
        axes[0, 1].set_ylabel("loss")
        axes[0, 1].legend()
        axes[0, 1].grid()

        # =========================
        # (3) Classification loss
        # =========================
        axes[1, 0].plot(cls_losses, label="cls loss")
        axes[1, 0].set_title("Classification Loss")
        axes[1, 0].set_xlabel("epoch")
        axes[1, 0].set_ylabel("loss")
        axes[1, 0].legend()
        axes[1, 0].grid()

        # =========================
        # (4) Gender classification loss
        # =========================
        axes[1, 1].plot(cls_gender_losses, label="gender cls")
        axes[1, 1].set_title("Gender Classification Loss")
        axes[1, 1].set_xlabel("epoch")
        axes[1, 1].set_ylabel("loss")
        axes[1, 1].legend()
        axes[1, 1].grid()

        # =========================
        # (5) L1 loss
        # =========================
        axes[2, 0].plot(l1_losses, label="L1 loss")
        axes[2, 0].set_title("L1 Regularization")
        axes[2, 0].set_xlabel("epoch")
        axes[2, 0].set_ylabel("loss")
        axes[2, 0].legend()
        axes[2, 0].grid()

        # =========================
        # (6) Mutual Information loss
        # =========================
        axes[2, 1].plot(mi_losses, label="MI loss")
        axes[2, 1].set_title("Mutual Information Loss")
        axes[2, 1].set_xlabel("epoch")
        axes[2, 1].set_ylabel("MI loss")
        axes[2, 1].legend()
        axes[2, 1].grid()

        # =========================
        # (7) MI estimate
        # =========================
        axes[3, 0].plot(mi_estimates, label="MI estimate")
        axes[3, 0].set_title("Mutual Information Estimate")
        axes[3, 0].set_xlabel("epoch")
        axes[3, 0].set_ylabel("MI")
        axes[3, 0].legend()
        axes[3, 0].grid()

        # =========================
        # (8) Optional: loss gap (generalization gap)
        # =========================
        axes[3, 1].plot(sub_pred_m_mean, label="mean")
        axes[3, 1].plot(sub_pred_m_std, label="std")

        axes[3, 1].set_title("sub_pred_m")
        axes[3, 1].legend()
        axes[3, 1].grid()


        axes[3, 2].plot(sub_pred_p_mean, label="mean")
        axes[3, 2].plot(sub_pred_p_std, label="std")

        axes[3, 2].set_title("sub_pred_p")
        axes[3, 2].legend()
        axes[3, 2].grid()

        # val_arr = np.array(val_losses)
        # train_arr = np.array(train_losses)

        # if len(val_arr) == len(train_arr):
        #     gap = val_arr - train_arr
        #     axes[3, 1].plot(gap, label="val - train")
        # else:
        #     axes[3, 1].text(0.3, 0.5, "No aligned data", fontsize=12)

        # axes[3, 1].set_title("Generalization Gap")
        # axes[3, 1].set_xlabel("epoch")
        # axes[3, 1].set_ylabel("gap")
        # axes[3, 1].legend()
        # axes[3, 1].grid()

        # =========================
        # layout
        # =========================
        plt.tight_layout()
        plt.savefig(os.path.join(dir, "loss_curves_4x2.png"), dpi=150)
        plt.close()

        #fig, axes = plt.subplots(2, 1, figsize=(8, 6))

        # axes[0].plot(train_losses, label="train")
        # axes[0].plot(rec_losses, label="Reconstruction Loss")
        # axes[0].plot(cls_losses, label="Classification Loss")
        # axes[0].plot(val_losses, label="val")

        # axes[0].set_xlabel("epoch")
        # axes[0].set_ylabel("loss")
        # axes[0].legend()
        # axes[0].set_title("Training Loss")

        # axes[1].plot(mi_losses, label="MI_loss")
        # axes[1].plot(mi_estimates, label="MI Estimate")

        # axes[1].set_xlabel("epoch")
        # axes[1].set_ylabel("MI")
        # axes[1].legend()
        # axes[1].set_title("Mutual Information")

        # plt.tight_layout()

        # os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)
        # plt.savefig(plot_path, dpi=150)
        # plt.close()

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

    train_mri.to_csv("train_mri.csv", index=False)
    val_mri.to_csv("val_mri.csv", index=False)
    test_mri.to_csv("test_mri.csv", index=False)
    
    train_pet.to_csv("train_pet.csv", index=False)
    val_pet.to_csv("val_pet.csv", index=False)
    test_pet.to_csv("test_pet.csv", index=False)

    # get scalers from train
    label_encoder = train_dataset.label_encoder
    gender_encoder = train_dataset.gender_encoder
    mri_scaler = train_dataset.mri_scaler
    pet_scaler = train_dataset.pet_scaler

    val_dataset = ROIDataset(
        val_mri,
        val_pet,
        mri_scaler=mri_scaler,
        pet_scaler=pet_scaler,
        label_encoder=label_encoder,
        gender_encoder=gender_encoder,
        fit=False
    )

    test_dataset = ROIDataset(
        test_mri,
        test_pet,
        mri_scaler=mri_scaler,
        pet_scaler=pet_scaler,
        label_encoder=label_encoder,
        gender_encoder=gender_encoder,
        fit=False
    )

    # model training
    model = AE(num_rois=162, latent=128, dim=512, beta_mi=0.1, beta_ib=0.1, l1_lambda=0.001, beta_ib_gender = 0.1,dropout_p=0.1, num_classes=len(label_encoder.classes_))
    club = CLUB(x_dim=128, y_dim=128//32, hidden_size=128).to(device)

    best_val_loss, best_model = train_model(model, club, train_dataset, val_dataset, device, epochs = 2000, lr = 1e-5, batch_size = 8, plot_path=os.path.join(dir, "loss_curve.png"))

    torch.save(best_model, f"{dir}/best.pth")
    print("Best Val Loss:", best_val_loss)

    # test evaluation
    model.load_state_dict(best_model)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        total_loss = 0
        total_n = 0
        for idx, (mri, pet, y, y_gender) in enumerate(test_loader):
            mri = mri.to(device)
            pet = pet.to(device)
            y = y.to(device)
            y_gender = y_gender.to(device)

            out = model(mri, pet, y, y_gender)

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