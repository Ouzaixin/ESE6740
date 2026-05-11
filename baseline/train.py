import os
import random
import copy
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from model import AE, CLUB
from dataset import ROIDataset
from utils import visualize

class LinearProbe(nn.Module):
    """
    Diagnostic classifier for z_{mod}.
    This is trained on detached z_{mod} so it does not affect the encoder.
    It only measures in some sense the degree of disentanglement.
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, z):
        return self.fc(z)
    
def batch_accuracy(logits, y):
    preds = torch.argmax(logits, dim = 1)
    return (preds == y).float().mean()

def prediction_entropy(logits, eps=1e-8):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return entropy.mean()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def normalize_latent(z, eps=1e-6):
    """
    Batch-normalize a latent only for CLUB estimation/training.
    This does not change the latents used by the decoder/classifier.
    """
    return (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + eps)


def train_model(
    model,
    club,
    train_dataset,
    val_dataset,
    device,
    epochs=2000,
    lr=5e-4,
    club_lr=None,
    lambda_apoe = 0.5,
    lambda_age = 0.05,
    probe_lr=1e-3,
    batch_size=64,
    club_steps=1,
    club_warmup_epochs=20,
    adv_warmup_epochs=20,
    adv_weight=1e-2,
    normalize_club_latents=True,
    print_acc_every=20,
    plot_path="loss_curve.png",
):
    """
    Alternating training:
      1. Update CLUB on detached latents from the current model.
      2. Update AE using ReLU(CLUB estimate) as the MI penalty.

    Stabilization details:
      - CLUB uses batch-normalized latents by default.
      - CLUB penalty is disabled during warmup epochs.
      - Checkpoint selection uses validation AE loss only.

    The model checkpoint is selected by validation AE loss only, not by CLUB,
    because CLUB estimates can be noisy and should not dominate model selection.
    """
    if club_lr is None:
        club_lr = lr

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    club.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    opt_club = torch.optim.Adam(club.parameters(), lr=club_lr)

    # Diagnostic classifier for z_mod -> y
    # This is NOT part of the main objective
    num_classes = model.cls_sub[-1].out_features
    z_mod_dim = model.partition.mod.out_features

    mod_probe = LinearProbe(in_dim=z_mod_dim, num_classes=num_classes).to(device)
    opt_probe = torch.optim.Adam(mod_probe.parameters(), lr = probe_lr)

    best_val_loss = float("inf")
    best_model = None

    train_objective_losses = []
    train_ae_losses = []
    rec_mri_losses = []
    rec_pet_losses = []
    rec_losses = []
    cls_losses = []
    val_losses = []
    mi_losses = []
    mi_estimates = []
    mi_penalties = []
    adv_losses = []
    probe_losses = []
    mod_entropies = []
    train_zs_accs = []
    train_zm_accs = []
    val_zs_accs = []
    val_zm_accs = []

    os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)

    for ep in range(epochs):
        total_train_objective = 0.0
        total_train_ae_loss = 0.0
        total_rec_mri = 0.00
        total_rec_pet = 0.00
        total_rec_loss = 0.0
        total_cls_loss = 0.0
        total_mi_loss = 0.0
        total_mi_estimate = 0.0
        total_mi_penalty = 0.0
        total_adv_loss = 0.0
        total_probe_loss = 0.0
        total_mod_entropy = 0.0
        total_zs_acc = 0.0
        total_zm_acc = 0.0

        # for mri, pet, y in train_loader:
        for mri, pet, y, apoe, age in train_loader:
            mri = mri.to(device)
            pet = pet.to(device)
            apoe = apoe.to(device)
            age = age.to(device)
            y = y.to(device)
            batch_n = mri.size(0)
            
            # ==========================================================
            # Step 1: update CLUB first, using detached current latents.
            # ==========================================================
            model.train()
            club.train()
            set_requires_grad(model, False)
            set_requires_grad(club, True)

            mi_loss_accum = 0.0
            for _ in range(club_steps):
                # Recompute latents for the current minibatch. We do not
                # backpropagate through the AE in the CLUB update.
                with torch.no_grad():
                    out_club = model(mri, pet, y)
                    z_mod_detached = out_club["z_mod"].detach()
                    z_sub_detached = out_club["z_sub"].detach()

                    if normalize_club_latents:
                        z_mod_detached = normalize_latent(z_mod_detached)
                        z_sub_detached = normalize_latent(z_sub_detached)

                mi_loss = club.learning_loss(z_mod_detached, z_sub_detached)
                opt_club.zero_grad()
                mi_loss.backward()
                opt_club.step()
                mi_loss_accum += mi_loss.item()

            mi_loss_avg = mi_loss_accum / max(club_steps, 1)

            # ==========================================================
            # Step 3: update AE using CLUB + adversarial confusion.
            # ==========================================================
            model.train()
            club.eval()
            set_requires_grad(model, True)
            set_requires_grad(club, False)

            out = model(mri, pet, y)

            model.train()
            mod_probe.train()

            set_requires_grad(model, False)
            set_requires_grad(mod_probe, True)

            with torch.no_grad():
                out_probe = model(mri, pet, y)
                z_mod_for_probe = out_probe["z_mod"].detach()

            logits_mod_probe = mod_probe(z_mod_for_probe)
            probe_loss = F.cross_entropy(logits_mod_probe, y)

            opt_probe.zero_grad()
            probe_loss.backward()
            opt_probe.step()

            # Raw CLUB estimate can be slightly negative due to minibatch noise.
            # Use ReLU for the penalty so the model is not rewarded for pushing
            # the estimator below zero.
            z_mod_for_club = out["z_mod"]
            z_sub_for_club = out["z_sub"]

            if normalize_club_latents:
                z_mod_for_club = normalize_latent(z_mod_for_club)
                z_sub_for_club = normalize_latent(z_sub_for_club)

            mi_raw = club(z_mod_for_club, z_sub_for_club)
            mi_penalty = torch.relu(mi_raw)

            # Warmup: train and log CLUB, but do not let it affect AE yet.
            if ep < club_warmup_epochs:
                mi_weight = 0.0
            else:
                mi_weight = model.beta_mi

            # loss = out["loss"] + mi_weight * mi_penalty

            # Disease adversarial loss on z_mod.
            # Freeze the probe, but allow gradients to flow through z_mod into the AE.
            set_requires_grad(mod_probe, False)

            logits_mod_adv = mod_probe(out["z_mod"])
            mod_entropy = prediction_entropy(logits_mod_adv)

            # Minimize negative entropy = maximize entropy.
            # This makes y hard to predict from z_mod.
            adv_loss = -mod_entropy

            if ep < adv_warmup_epochs:
                current_adv_weight = 0.0
            else:
                # current_adv_weight = adv_weight
                # current_adv_weight = adv_weight * min(1.0, (ep - adv_warmup_epochs) / 100)
                current_adv_weight = adv_weight * min(1.0, (ep - adv_warmup_epochs) / 100)
            loss_apoe = F.cross_entropy(out["apoe_pred"], apoe.long())
            loss_age = F.mse_loss(out["age_pred"], age.float())
            # lambda_apoe = 0.5
            # lambda_age = 0.05

            set_requires_grad(model, True)
            set_requires_grad(club, False)
            set_requires_grad(mod_probe, False)

            # loss = out["loss"] + mi_weight * mi_penalty + current_adv_weight * adv_loss
            loss = (
                out["loss"]
                + lambda_apoe * loss_apoe
                + lambda_age * loss_age
                + mi_weight * mi_penalty
                + current_adv_weight * adv_loss
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            set_requires_grad(mod_probe, True)
            # diagnostic classification accuracies
            # mod_probe.train()

            # z_mod_detached_for_probe = out["z_mod"].detach()
            # logits_mod_probe = mod_probe(z_mod_detached_for_probe)
            # probe_loss = F.cross_entropy(logits_mod_probe, y)

            # opt_probe.zero_grad()
            # probe_loss.backward()
            # opt_probe.step()

            with torch.no_grad():
                # Accuracy using z_s
                logits_sub = model.cls_sub(out["z_sub"])
                logits_mod = mod_probe(out["z_mod"])

                acc_zs = batch_accuracy(logits_sub, y)
                # Accuracy using z_m
                logits_mod = mod_probe(out["z_mod"].detach())
                acc_zm = batch_accuracy(logits_mod, y)

            total_zs_acc += acc_zs.item() * batch_n
            total_zm_acc += acc_zm.item() * batch_n            


            total_train_objective += loss.item() * batch_n
            total_train_ae_loss += out["loss"].item() * batch_n
            total_rec_mri += out["loss_rec_mri"].item() * batch_n
            total_rec_pet += out["loss_rec_pet"].item() * batch_n
            total_rec_loss += out["loss_rec"].item() * batch_n
            total_cls_loss += out["loss_cls"].item() * batch_n
            total_mi_loss += mi_loss_avg * batch_n
            total_mi_estimate += mi_raw.item() * batch_n
            total_mi_penalty += mi_penalty.item() * batch_n
            total_adv_loss += adv_loss.item() * batch_n
            total_probe_loss += probe_loss.item() * batch_n
            total_mod_entropy += mod_entropy.item() * batch_n
        # Restore gradients for both modules before validation/next epoch.
        set_requires_grad(model, True)
        set_requires_grad(club, True)
        total_zs_acc /= len(train_dataset)
        total_zm_acc /= len(train_dataset)
        train_zs_accs.append(total_zs_acc)
        train_zm_accs.append(total_zm_acc)

        total_train_objective /= len(train_dataset)
        total_train_ae_loss /= len(train_dataset)
        total_rec_mri /= len(train_dataset)
        total_rec_pet /= len(train_dataset)
        total_rec_loss /= len(train_dataset)
        total_cls_loss /= len(train_dataset)
        total_mi_loss /= len(train_dataset)
        total_mi_estimate /= len(train_dataset)
        total_mi_penalty /= len(train_dataset)
        total_adv_loss /= len(train_dataset)
        total_probe_loss /= len(train_dataset)
        total_mod_entropy /= len(train_dataset)

        train_objective_losses.append(total_train_objective)
        train_ae_losses.append(total_train_ae_loss)
        rec_mri_losses.append(total_rec_mri)
        rec_pet_losses.append(total_rec_pet)
        rec_losses.append(total_rec_loss)
        cls_losses.append(total_cls_loss)
        mi_losses.append(total_mi_loss)
        mi_estimates.append(total_mi_estimate)
        mi_penalties.append(total_mi_penalty)
        adv_losses.append(total_adv_loss)
        probe_losses.append(total_probe_loss)
        mod_entropies.append(total_mod_entropy)

        # -------------------------
        # Validation: use AE loss only for model selection.
        # -------------------------
        model.eval()
        club.eval()
        val_zs_acc = 0.0
        val_zm_acc = 0.0
        mod_probe.eval()
        val_loss = 0.0

        with torch.no_grad():
            for mri, pet, y, apoe, age in val_loader:
                mri = mri.to(device)
                pet = pet.to(device)
                apoe = apoe.to(device)
                age = age.to(device)
                y = y.to(device)

                out = model(mri, pet, y)
                loss = out["loss"]
                val_loss += loss.item() * mri.size(0)
                loss_apoe = F.cross_entropy(out["apoe_pred"], apoe.long())
                loss_age = F.mse_loss(out["age_pred"], age.float())
                logits_sub = model.cls_sub(out["z_sub"])
                logits_mod = mod_probe(out["z_mod"])

                val_zs_acc += batch_accuracy(logits_sub, y).item() * mri.size(0)
                val_zm_acc += batch_accuracy(logits_mod, y).item() * mri.size(0)

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        val_zs_acc /= len(val_dataset)
        val_zm_acc /= len(val_dataset)

        val_zs_accs.append(val_zs_acc)
        val_zm_accs.append(val_zm_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

        # -------------------------
        # Plot curves.
        # -------------------------
        plt.figure(figsize=(9, 5))
        plt.plot(train_objective_losses, label="Train objective: AE + scheduled beta_mi * ReLU(MI)")
        plt.plot(train_ae_losses, label="Train AE loss")
        plt.plot(rec_mri_losses, label = "Reconstruction MRI")
        plt.plot(rec_pet_losses, label = "Reconstruction PET")
        plt.plot(rec_losses, label="Reconstruction loss")
        plt.plot(cls_losses, label="Classification loss")
        plt.plot(val_losses, label="Validation AE loss")
        plt.plot(mi_losses, label="CLUB learning loss")
        plt.plot(mi_estimates, label="Raw CLUB estimate")
        plt.plot(mi_penalties, label="ReLU CLUB penalty")
        plt.plot(adv_losses, label="Adv loss: -H(y|z_m)")
        plt.plot(probe_losses, label="z_m probe CE loss")
        plt.plot(mod_entropies, label="Entropy of probe on z_m")
        
        plt.xlabel("epoch")
        plt.ylabel("loss / estimate")
        plt.legend(fontsize=8)
        plt.title("Training Curve")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # Accuracy plot
        plt.figure(figsize=(9,5))
        plt.plot(train_zs_accs, label ="Train acc using z_s")
        plt.plot(train_zm_accs, label ="Train acc using z_m")
        plt.plot(val_zs_accs, label ="Val acc using z_s")
        plt.plot(val_zm_accs, label ="Val acc using z_m")        
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(fontsize=8)
        plt.title("Disentanglement Curve")
        plt.tight_layout()
        acc_plot_path = os.path.join(os.path.dirname(plot_path), "accuracy_curve.png")
        plt.savefig(acc_plot_path, dpi=150)
        plt.close()

        if ep % print_acc_every == 0:
            print(
                f"Epoch {ep + 1}/{epochs} | "
                f"TrainObj {total_train_objective:.4f} | "
                f"TrainAE {total_train_ae_loss:.4f} | "
                f"Rec {total_rec_loss:.4f} | "
                f"Cls {total_cls_loss:.4f} | "
                f"ValAE {val_loss:.4f} | "
                f"CLUB-Loss {total_mi_loss:.4f} | "
                f"MI-Raw {total_mi_estimate:.4f} | "
                f"MI-Penalty {total_mi_penalty:.4f} | "
                f"Train Acc z_s {total_zs_acc:.4f} | "
                f"Train Acc z_m {total_zm_acc:.4f} | "
                f"Val Acc z_s {val_zs_acc:.4f} | "
                f"Val Acc z_m {val_zm_acc:.4f} | "
                f"CLUB-active {ep >= club_warmup_epochs}"
                f"ProbeCE {total_probe_loss:.4f} | "
                f"AdvLoss {total_adv_loss:.4f} | "
                f"Entropy_zm {total_mod_entropy:.4f} | "
                f"Adv-active {ep >= adv_warmup_epochs} | "
            )

    history = {
        "train_objective_losses": train_objective_losses,
        "train_ae_losses": train_ae_losses,
        "rec_mri_losses": rec_mri_losses,
        "rec_pet_losses": rec_pet_losses,
        "rec_losses": rec_losses,
        "cls_losses": cls_losses,
        "val_losses": val_losses,
        "mi_losses": mi_losses,
        "mi_estimates": mi_estimates,
        "mi_penalties": mi_penalties,
        "adv_losses": adv_losses,
        "probe_losses": probe_losses,
        "mod_entropies": mod_entropies,
        "train_zs_accs": train_zs_accs,
        "train_zm_accs": train_zm_accs,
        "val_zs_accs": val_zs_accs,
        "val_zm_accs": val_zm_accs,
    }
    history_path = os.path.join(os.path.dirname(plot_path), "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    return best_val_loss, best_model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    mri_df = pd.read_csv("./dataset/MRI_CN_AD.csv")
    pet_df = pd.read_csv("./dataset/PET_CN_AD.csv")
    # mri_df = pd.read_csv("./dataset/MRI_roi.csv")
    # pet_df = pd.read_csv("./dataset/PET_roi.csv")

    indices = mri_df.index.values

    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_mri = mri_df.iloc[train_idx].reset_index(drop=True)
    val_mri = mri_df.iloc[val_idx].reset_index(drop=True)
    test_mri = mri_df.iloc[test_idx].reset_index(drop=True)

    train_pet = pet_df.iloc[train_idx].reset_index(drop=True)
    val_pet = pet_df.iloc[val_idx].reset_index(drop=True)
    test_pet = pet_df.iloc[test_idx].reset_index(drop=True)

    train_dataset = ROIDataset(train_mri, train_pet, fit=True)
    print("MRI missing rate:", train_dataset.mri_missing_rate_before_impute)
    print("PET missing rate:", train_dataset.pet_missing_rate_before_impute)
    print("MRI scaled mean/std:", train_dataset.X_mri.mean(), train_dataset.X_mri.std())
    print("PET scaled mean/std:", train_dataset.X_pet.mean(), train_dataset.X_pet.std())
    
    label_encoder = train_dataset.label_encoder
    mri_imputer = train_dataset.mri_imputer
    pet_imputer = train_dataset.pet_imputer
    mri_scaler = train_dataset.mri_scaler
    pet_scaler = train_dataset.pet_scaler
    age_mean = train_dataset.age_mean
    age_std = train_dataset.age_std
    apoe_fill = train_dataset.apoe_fill
    with open(os.path.join(results_dir, "preprocess_objects.pkl"), "wb") as f:
        pickle.dump(
            {
                "mri_imputer": mri_imputer,
                "pet_imputer": pet_imputer,
                "mri_scaler": mri_scaler,
                "pet_scaler": pet_scaler,
                "label_encoder": label_encoder,
                "age_mean": age_mean,
                "age_std": age_std,
                "apoe_fill": apoe_fill,
            },
            f,
        )

    val_dataset = ROIDataset(
        val_mri,
        val_pet,
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
    test_dataset = ROIDataset(
        test_mri,
        test_pet,
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

    # Infer ROI count instead of hard-coding 162.
    num_rois = train_dataset.X_mri.shape[1]
    latent = 128
    subtype_dim = latent // 16
    # subtype_dim = 16
    # lambda_apoe = 0.25
    # lambda_age = 0.05
    # lambda_apoe = 0.5
    lambda_apoe = 0.0
    lambda_age = 0.0
    print(f"Detected num_rois = {num_rois}")
    print(f"Label classes = {list(label_encoder.classes_)}")
    print(f"Subtype latent dim = {subtype_dim}")

    model = AE(
        num_rois=num_rois,
        latent=latent,
        dim=512,
        subtype_dim=subtype_dim,
        beta_mi=0.0,   
        beta_ib=0.0,
        dropout_p=0.1,
        num_classes=len(label_encoder.classes_),
    )

    # Current training uses club(z_mod, z_sub), i.e. q(z_sub | z_mod).
    # club = CLUB(x_dim=latent, y_dim=subtype_dim, hidden_size=128).to(device)
    club = CLUB(x_dim=latent, y_dim=subtype_dim, hidden_size=256).to(device)

    best_val_loss, best_model = train_model(
        model,
        club,
        train_dataset,
        val_dataset,
        device,
        epochs = 400,
        lr=5e-4,
        club_lr=1e-5,
        probe_lr=1e-3,
        batch_size=512,
        club_steps=5,
        lambda_apoe = lambda_apoe,
        lambda_age = lambda_age,
        club_warmup_epochs=50,
        adv_warmup_epochs=50,
        adv_weight=0,
        normalize_club_latents=True,
        print_acc_every=50,
        plot_path=os.path.join(results_dir, "loss_curve.png"),
    )
    torch.save(best_model, os.path.join(results_dir, "best.pth"))
    print("Best Val Loss:", best_val_loss)

    model.load_state_dict(best_model)
    model.to(device)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.load_state_dict(best_model)
    model.to(device)
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        total_ae_loss = 0.0
        total_rec_loss = 0.0
        total_rec_mri = 0.0
        total_rec_pet = 0.0
        total_cls_loss = 0.0
        total_apoe_loss = 0.0
        total_age_loss = 0.0
        total_n = 0
        
        for idx, (mri, pet, y, apoe, age) in enumerate(test_loader):
            mri = mri.to(device)
            pet = pet.to(device)
            y = y.to(device)
            apoe = apoe.to(device)
            age = age.to(device)

            out = model(mri, pet, y)

            loss_apoe = F.cross_entropy(out["apoe_pred"], apoe.long())
            loss_age = F.mse_loss(out["age_pred"], age.float())

            # Test objective matching the supervised clinical part.
            # This excludes CLUB/adversarial terms because those are training-time regularizers.
            test_loss = (
                out["loss"]
                + lambda_apoe * loss_apoe
                + lambda_age * loss_age
            )

            if idx == 0:
                for i in range(min(5, mri.size(0))):
                    visualize(
                        mri[i],
                        pet[i],
                        out["reconstruction_m"][i],
                        out["reconstruction_p"][i],
                        save_path=os.path.join(results_dir, f"test_sample_{i + 1}.png"),
                    )

            batch_n = mri.size(0)

            total_loss += test_loss.item() * batch_n
            total_ae_loss += out["loss"].item() * batch_n
            total_rec_loss += out["loss_rec"].item() * batch_n
            total_rec_mri += out["loss_rec_mri"].item() * batch_n
            total_rec_pet += out["loss_rec_pet"].item() * batch_n
            total_cls_loss += out["loss_cls"].item() * batch_n
            total_apoe_loss += loss_apoe.item() * batch_n
            total_age_loss += loss_age.item() * batch_n
            total_n += batch_n

        print(
            f"Test Total {total_loss / total_n:.4f} | "
            f"AE {total_ae_loss / total_n:.4f} | "
            f"Rec {total_rec_loss / total_n:.4f} | "
            f"RecMRI {total_rec_mri / total_n:.4f} | "
            f"RecPET {total_rec_pet / total_n:.4f} | "
            f"DXCls {total_cls_loss / total_n:.4f} | "
            f"APOE {total_apoe_loss / total_n:.4f} | "
            f"Age {total_age_loss / total_n:.4f}"
        )


# import torch
# import math
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.manifold import TSNE
# import pandas as pd
# import numpy as np
# import os
# import matplotlib
# import matplotlib.pyplot as plt
# import random
# import copy 
# from sklearn.model_selection import train_test_split

# from model import AE, CLUB
# from dataset import ROIDataset
# from utils import visualize

# import pickle

# matplotlib.use("Agg")

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def train_model(model, club, train_dataset, val_dataset, device, epochs=100, lr=1e-4, batch_size=64, plot_path="loss_curve.png"):

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     club.to(device)
#     opt_club = torch.optim.Adam(club.parameters(), lr=lr)

#     best_val_loss = float("inf")
#     best_model = None

#     train_losses = []
#     rec_losses = []
#     cls_losses = []
#     val_losses = []
#     mi_losses = []
#     mi_estimates = []

#     plt.figure()

#     for ep in range(epochs):

#         total_train_loss = 0
#         total_rec_loss = 0
#         total_cls_loss = 0
#         total_mi_loss = 0
#         mi_estimate = 0

#         for mri, pet, y in train_loader:
#             model.train()
#             club.eval()
#             mri = mri.to(device)
#             pet = pet.to(device)
#             y = y.to(device)

#             out = model(mri, pet, y)
#             mi_raw = club(out["z_mod"], out["z_sub"])
#             mi_penalty = torch.relu(mi_raw)
#             # loss = out["loss"] + model.beta_mi * mi
#             loss = out["loss"] + model.beta_mi * mi_penalty
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             for _ in range(10):
#                 club.train()
#                 mi_loss = club.learning_loss(out["z_mod"].detach(), out["z_sub"].detach())
#                 opt_club.zero_grad()
#                 mi_loss.backward()
#                 opt_club.step()

#             total_train_loss += out["loss"].item() * mri.size(0)
#             total_rec_loss += out["loss_rec"].item() * mri.size(0)
#             total_cls_loss += out["loss_cls"].item() * mri.size(0)
            
#             total_mi_loss += mi_loss.item() * mri.size(0)
#             # mi_estimate += mi.item() * mri.size(0)
#             mi_estimate += mi_raw.item() * mri.size(0)

#         total_train_loss /= len(train_dataset)
#         total_rec_loss /= len(train_dataset)
#         total_cls_loss /= len(train_dataset)

#         mi_estimate /= len(train_dataset)
#         total_mi_loss /= len(train_dataset)

#         train_losses.append(total_train_loss)
#         rec_losses.append(total_rec_loss)
#         cls_losses.append(total_cls_loss)
#         mi_losses.append(total_mi_loss)
#         mi_estimates.append(mi_estimate)

#         # validation
#         model.eval()
#         val_loss = 0

#         with torch.no_grad():
#             for mri, pet, y in val_loader:
#                 mri = mri.to(device)
#                 pet = pet.to(device)
#                 y = y.to(device)
#                 out = model(mri, pet, y)
#                 loss = out["loss"]
#                 val_loss += loss.item() * mri.size(0)

#         val_loss /= len(val_dataset)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             # best_model = model.state_dict()
#             best_model = copy.deepcopy(model.state_dict())
#             # visualize(
#             #     mri[0],
#             #     pet[0],
#             #     out["reconstruction_m"][0],
#             #     out["reconstruction_p"][0],
#             #     save_path=os.path.join(dir, f"val_epoch_{ep+1}.png")
#             # )

#         plt.clf()
#         plt.plot(train_losses, label="train")
#         plt.plot(rec_losses, label="Reconstruction Loss")
#         plt.plot(cls_losses, label="Classification Loss")
#         plt.plot(val_losses, label="val")
#         plt.plot(mi_losses, label="MI_loss")
#         plt.plot(mi_estimates, label="MI Estimate")

#         plt.xlabel("epoch")
#         plt.ylabel("loss")
#         plt.legend()
#         plt.title("Training Curve")

#         os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)
#         plt.savefig(plot_path, dpi=150)
#         plt.close()

#         print(f"Epoch {ep+1}/{epochs} | Train {total_train_loss:.4f} | Rec_Loss {total_rec_loss:.4f} | Cls_Loss {total_cls_loss:.4f} | Val {val_loss:.4f} | MI_Loss {total_mi_loss:.4f} | MI_Estimate {mi_estimate:.4f} |")

#     return best_val_loss, best_model

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     set_seed(42)

#     dir = "./results"
#     os.makedirs(dir, exist_ok=True)

#     mri_df = pd.read_csv("./dataset/MRI_CN_AD.csv")
#     pet_df = pd.read_csv("./dataset/PET_CN_AD.csv")

#     indices = mri_df.index.values

#     train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
#     val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

#     train_mri = mri_df.iloc[train_idx].reset_index(drop=True)
#     val_mri   = mri_df.iloc[val_idx].reset_index(drop=True)
#     test_mri  = mri_df.iloc[test_idx].reset_index(drop=True)

#     train_pet = pet_df.iloc[train_idx].reset_index(drop=True)
#     val_pet   = pet_df.iloc[val_idx].reset_index(drop=True)
#     test_pet  = pet_df.iloc[test_idx].reset_index(drop=True)

#     train_dataset = ROIDataset(train_mri, train_pet, fit=True)

#     # get scalers from train
#     label_encoder = train_dataset.label_encoder
#     mri_scaler = train_dataset.mri_scaler
#     pet_scaler = train_dataset.pet_scaler

#     pickle.dump({
#         "mri_scaler": mri_scaler,
#         "pet_scaler": pet_scaler,
#         "label_encoder": label_encoder,
#     }, open("./results/preprocess_objects.pkl", "wb"))

#     val_dataset = ROIDataset(
#         val_mri,
#         val_pet,
#         mri_scaler=mri_scaler,
#         pet_scaler=pet_scaler,
#         label_encoder=label_encoder,
#         fit=False
#     )

#     test_dataset = ROIDataset(
#         test_mri,
#         test_pet,
#         mri_scaler=mri_scaler,
#         pet_scaler=pet_scaler,
#         label_encoder=label_encoder,
#         fit=False
#     )

#     # model training
#     model = AE(num_rois=162, latent=128, dim=512, beta_mi=0.1, beta_ib=0.1, dropout_p=0.0, num_classes=len(label_encoder.classes_))
#     club = CLUB(x_dim=128, y_dim=128//32, hidden_size=128).to(device)

#     best_val_loss, best_model = train_model(model, club, train_dataset, val_dataset, device, epochs = 10000, lr = 1e-4, batch_size = 64, plot_path=os.path.join(dir, "loss_curve.png"))

#     torch.save(best_model, f"{dir}/best.pth")
#     print("Best Val Loss:", best_val_loss)

#     # test evaluation
#     model.load_state_dict(best_model)
#     model.eval()
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#     with torch.no_grad():
#         total_loss = 0
#         total_n = 0
#         for idx, (mri, pet, y) in enumerate(test_loader):
#             mri = mri.to(device)
#             pet = pet.to(device)
#             y = y.to(device)

#             out = model(mri, pet, y)

#             if idx == 0:
#                 for i in range(min(5, mri.size(0))):
#                     visualize(
#                         mri[i],
#                         pet[i],
#                         out["reconstruction_m"][i],
#                         out["reconstruction_p"][i],
#                         save_path=os.path.join(dir, f"test_epoch_{i+1}.png")
#                     )

#             batch_size = mri.size(0)
#             total_loss += out["loss"].item() * batch_size
#             total_n += batch_size

#         print("Test Loss:", total_loss / total_n)

