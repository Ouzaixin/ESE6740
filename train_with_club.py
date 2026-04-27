import os
import copy
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from model import AE
from dataset import ROIDataset
from utils import visualize
from club import CLUBForAD, set_requires_grad


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_float(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)


def update_running_sums(running, metrics, batch_size):
    """
    Accumulate scalar metrics over a full epoch.
    """
    for key in running.keys():
        if key in metrics and metrics[key] is not None:
            running[key] += _to_float(metrics[key]) * batch_size


def normalize_running_sums(running, n_samples):
    """
    Convert accumulated sums into dataset averages.
    """
    return {key: value / max(n_samples, 1) for key, value in running.items()}


def make_running_dict():
    return {
        "loss_total": 0.0,
        "loss_ae": 0.0,
        "loss_rec": 0.0,
        "loss_cls": 0.0,
        "loss_dis": 0.0,
        "loss_kl": 0.0,
        "loss_club_mi": 0.0,
        "club_mi_m": 0.0,
        "club_mi_p": 0.0,
        "club_train_nll": 0.0,
        "club_ll_m": 0.0,
        "club_ll_p": 0.0,
    }


def collect_batch_metrics(out, loss_total, loss_club_mi, mi_m, mi_p, club_train_nll=None, club_ll=None):
    """
    Collect all batch-level metrics into a plain dictionary.
    """
    metrics = {
        "loss_total": loss_total,
        "loss_ae": out["loss"],
        "loss_rec": out.get("loss_rec"),
        "loss_cls": out.get("loss_cls"),
        "loss_dis": out.get("loss_dis"),
        "loss_kl": out.get("loss_kl"),
        "loss_club_mi": loss_club_mi,
        "club_mi_m": mi_m,
        "club_mi_p": mi_p,
        "club_train_nll": club_train_nll,
    }

    if club_ll is not None:
        metrics["club_ll_m"] = club_ll.get("club_ll_m")
        metrics["club_ll_p"] = club_ll.get("club_ll_p")

    return metrics


def train_model_with_club(
    model,
    club,
    train_dataset,
    val_dataset,
    device,
    epochs=100,
    lr=1e-4,
    club_lr=1e-4,
    batch_size=64,
    lambda_club=0.001,
    club_sampled_train=True,
    club_sampled_val=False,
    club_steps_per_batch=1,
    plot_path="loss_curve_with_club.png",
    results_dir="results_club",
):
    """
    Alternating training loop:

    Step 1: update CLUB networks using detached latents.
    Step 2: update AE/VIB model using AE loss + lambda_club * CLUB MI estimate.

    Notes:
        - CLUB uses z_sub, not mu_sub.
        - During validation, model.eval() disables modality masking by default
          in the updated AE.forward(..., apply_mask=None).
        - CLUB modules are training-time regularizers. They are not needed
          for subtype inference, but they are saved for diagnostics/reproducibility.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.to(device)
    club.to(device)

    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_club = torch.optim.Adam(club.parameters(), lr=club_lr)

    best_val_loss = float("inf")
    best_model = None
    best_club = None

    train_losses = []
    val_losses = []

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(
        os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".",
        exist_ok=True,
    )

    for ep in range(epochs):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        club.train()

        train_running = make_running_dict()

        for mri, pet in train_loader:
            mri = mri.to(device)
            pet = pet.to(device)
            batch_n = mri.size(0)

            # ------------------------------------------------------------
            # Step 1: update CLUB networks only
            # ------------------------------------------------------------
            set_requires_grad(model, False)
            set_requires_grad(club, True)
            club.train()

            club_train_nll_value = None
            club_ll_value = None

            for _ in range(club_steps_per_batch):
                # We use no_grad because the AE is frozen in this step.
                # Latents are detached before going into CLUB.
                with torch.no_grad():
                    out_detached = model(mri, pet)
                    z_sub_detached = out_detached["z_sub"].detach()
                    z_m_detached = out_detached["z_m"].detach()
                    z_p_detached = out_detached["z_p"].detach()

                loss_club_train = club.learning_loss(
                    z_sub_detached,
                    z_m_detached,
                    z_p_detached,
                )

                optimizer_club.zero_grad()
                loss_club_train.backward()
                nn.utils.clip_grad_norm_(club.parameters(), max_norm=1.0)
                optimizer_club.step()

                club_train_nll_value = loss_club_train.detach()

            with torch.no_grad():
                club_ll_value = club.loglikelihood(
                    z_sub_detached,
                    z_m_detached,
                    z_p_detached,
                )

            # ------------------------------------------------------------
            # Step 2: update AE/VIB model only
            # ------------------------------------------------------------
            set_requires_grad(model, True)
            set_requires_grad(club, False)
            model.train()
            club.eval()

            out = model(mri, pet)

            loss_club_mi, mi_m, mi_p = club.mi_estimate(
                out["z_sub"],
                out["z_m"],
                out["z_p"],
                sampled=club_sampled_train,
            )

            loss_total = out["loss"] + lambda_club * loss_club_mi

            optimizer_model.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_model.step()

            batch_metrics = collect_batch_metrics(
                out=out,
                loss_total=loss_total,
                loss_club_mi=loss_club_mi,
                mi_m=mi_m,
                mi_p=mi_p,
                club_train_nll=club_train_nll_value,
                club_ll=club_ll_value,
            )
            update_running_sums(train_running, batch_metrics, batch_n)

        train_metrics = normalize_running_sums(train_running, len(train_dataset))
        train_losses.append(train_metrics["loss_total"])

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        club.eval()
        set_requires_grad(model, False)
        set_requires_grad(club, False)

        val_running = make_running_dict()
        last_val_batch = None

        with torch.no_grad():
            for mri, pet in val_loader:
                mri = mri.to(device)
                pet = pet.to(device)
                batch_n = mri.size(0)

                # In eval mode, the updated AE defaults to no modality masking.
                out = model(mri, pet)

                loss_club_mi, mi_m, mi_p = club.mi_estimate(
                    out["z_sub"],
                    out["z_m"],
                    out["z_p"],
                    sampled=club_sampled_val,
                )
                loss_total = out["loss"] + lambda_club * loss_club_mi

                club_ll = club.loglikelihood(
                    out["z_sub"],
                    out["z_m"],
                    out["z_p"],
                )

                # Validation does not train CLUB, so no train NLL here.
                batch_metrics = collect_batch_metrics(
                    out=out,
                    loss_total=loss_total,
                    loss_club_mi=loss_club_mi,
                    mi_m=mi_m,
                    mi_p=mi_p,
                    club_train_nll=None,
                    club_ll=club_ll,
                )
                update_running_sums(val_running, batch_metrics, batch_n)

                last_val_batch = (mri, pet, out)

        val_metrics = normalize_running_sums(val_running, len(val_dataset))
        val_losses.append(val_metrics["loss_total"])

        # Restore grad settings for next epoch.
        set_requires_grad(model, True)
        set_requires_grad(club, True)

        # -------------------------
        # Save best model + CLUB state
        # -------------------------
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            best_model = copy.deepcopy(model.state_dict())
            best_club = copy.deepcopy(club.state_dict())

            if last_val_batch is not None:
                mri, pet, out = last_val_batch

                visualize(
                    mri[0],
                    pet[0],
                    out["reconstruction_m"][0],
                    out["reconstruction_p"][0],
                    save_path=os.path.join(results_dir, f"val_epoch_{ep + 1}.png"),
                )

        # -------------------------
        # Plot training curve
        # -------------------------
        plt.figure()
        plt.plot(train_losses, label="train_total")
        plt.plot(val_losses, label="val_total")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Training Curve with CLUB")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(
            f"Epoch {ep + 1:03d}/{epochs} | "
            f"Train Total {train_metrics['loss_total']:.4f} "
            f"AE {train_metrics['loss_ae']:.4f} "
            f"Rec {train_metrics['loss_rec']:.4f} "
            f"Cls {train_metrics['loss_cls']:.4f} "
            f"Dis {train_metrics['loss_dis']:.4f} "
            f"KL {train_metrics['loss_kl']:.4f} "
            f"CLUB-MI {train_metrics['loss_club_mi']:.4f} "
            f"MI_m {train_metrics['club_mi_m']:.4f} "
            f"MI_p {train_metrics['club_mi_p']:.4f} "
            f"CLUB-NLL {train_metrics['club_train_nll']:.4f} | "
            f"Val Total {val_metrics['loss_total']:.4f} "
            f"AE {val_metrics['loss_ae']:.4f} "
            f"Rec {val_metrics['loss_rec']:.4f} "
            f"Cls {val_metrics['loss_cls']:.4f} "
            f"Dis {val_metrics['loss_dis']:.4f} "
            f"KL {val_metrics['loss_kl']:.4f} "
            f"CLUB-MI {val_metrics['loss_club_mi']:.4f} "
            f"MI_m {val_metrics['club_mi_m']:.4f} "
            f"MI_p {val_metrics['club_mi_p']:.4f}"
        )

    return best_val_loss, best_model, best_club


def evaluate_test_with_club(
    model,
    club,
    test_dataset,
    device,
    batch_size=64,
    lambda_club=0.001,
    club_sampled_test=False,
    results_dir="results_club",
):
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    club.eval()
    set_requires_grad(model, False)
    set_requires_grad(club, False)

    test_running = make_running_dict()
    total_n = 0

    with torch.no_grad():
        for idx, (mri, pet) in enumerate(test_loader):
            mri = mri.to(device)
            pet = pet.to(device)
            batch_n = mri.size(0)

            out = model(mri, pet)

            loss_club_mi, mi_m, mi_p = club.mi_estimate(
                out["z_sub"],
                out["z_m"],
                out["z_p"],
                sampled=club_sampled_test,
            )
            loss_total = out["loss"] + lambda_club * loss_club_mi

            club_ll = club.loglikelihood(
                out["z_sub"],
                out["z_m"],
                out["z_p"],
            )

            if idx == 0:
                n_vis = min(5, mri.size(0))
                for i in range(n_vis):
                    visualize(
                        mri[i],
                        pet[i],
                        out["reconstruction_m"][i],
                        out["reconstruction_p"][i],
                        save_path=os.path.join(results_dir, f"test_sample_{i + 1}.png"),
                    )

            batch_metrics = collect_batch_metrics(
                out=out,
                loss_total=loss_total,
                loss_club_mi=loss_club_mi,
                mi_m=mi_m,
                mi_p=mi_p,
                club_train_nll=None,
                club_ll=club_ll,
            )
            update_running_sums(test_running, batch_metrics, batch_n)
            total_n += batch_n

    test_metrics = normalize_running_sums(test_running, total_n)

    print(
        f"Test Total {test_metrics['loss_total']:.4f} "
        f"AE {test_metrics['loss_ae']:.4f} "
        f"Rec {test_metrics['loss_rec']:.4f} "
        f"Cls {test_metrics['loss_cls']:.4f} "
        f"Dis {test_metrics['loss_dis']:.4f} "
        f"KL {test_metrics['loss_kl']:.4f} "
        f"CLUB-MI {test_metrics['loss_club_mi']:.4f} "
        f"MI_m {test_metrics['club_mi_m']:.4f} "
        f"MI_p {test_metrics['club_mi_p']:.4f}"
    )

    set_requires_grad(model, True)
    set_requires_grad(club, True)

    return test_metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    curr_dir = os.getcwd()
    results_dir = os.path.join(curr_dir, "results_club")
    os.makedirs(results_dir, exist_ok=True)

    mri_df = pd.read_csv("./dataset/MRI_roi.csv")
    pet_df = pd.read_csv("./dataset/PET_roi.csv")

    indices = mri_df.index.values

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        shuffle=True,
    )

    train_mri = mri_df.iloc[train_idx].reset_index(drop=True)
    val_mri = mri_df.iloc[val_idx].reset_index(drop=True)
    test_mri = mri_df.iloc[test_idx].reset_index(drop=True)

    train_pet = pet_df.iloc[train_idx].reset_index(drop=True)
    val_pet = pet_df.iloc[val_idx].reset_index(drop=True)
    test_pet = pet_df.iloc[test_idx].reset_index(drop=True)

    train_dataset = ROIDataset(train_mri, train_pet, fit=True)

    # Get scalers from train only.
    mri_scaler = train_dataset.mri_scaler
    pet_scaler = train_dataset.pet_scaler

    val_dataset = ROIDataset(
        val_mri,
        val_pet,
        mri_scaler=mri_scaler,
        pet_scaler=pet_scaler,
        fit=False,
    )

    test_dataset = ROIDataset(
        test_mri,
        test_pet,
        mri_scaler=mri_scaler,
        pet_scaler=pet_scaler,
        fit=False,
    )

    # Infer ROI dimension from the processed dataset.
    n_rois = train_dataset.X_mri.shape[1]
    print(f"Detected n_rois = {n_rois}")

    latent_dim = 128

    # Start with lambda_dis=0.0 so CLUB is the main disentanglement penalty.
    # For a conservative first run, set lambda_club=0.0 to log CLUB only.
    lambda_club = 0.001

    model = AE(
        n_rois=n_rois,
        latent=latent_dim,
        hidden=512,
        lambda_cls=0.1,
        lambda_dis=0.0,
        beta_kl=0.01,
    )

    club = CLUBForAD(
        latent_dim=latent_dim,
        hidden_dim=128,
    )

    best_val_loss, best_model, best_club = train_model_with_club(
        model=model,
        club=club,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        epochs=200,
        lr=2e-4,
        club_lr=1e-4,
        batch_size=64,
        lambda_club=lambda_club,
        club_sampled_train=True,
        club_sampled_val=False,
        club_steps_per_batch=1,
        plot_path=os.path.join(results_dir, "loss_curve_with_club.png"),
        results_dir=results_dir,
    )

    best_model_path = os.path.join(results_dir, "best_model.pth")
    best_club_path = os.path.join(results_dir, "best_club.pth")
    checkpoint_path = os.path.join(results_dir, "best_checkpoint.pth")

    torch.save(best_model, best_model_path)
    torch.save(best_club, best_club_path)
    torch.save(
        {
            "model_state_dict": best_model,
            "club_state_dict": best_club,
            "best_val_loss": best_val_loss,
            "n_rois": n_rois,
            "latent_dim": latent_dim,
            "lambda_club": lambda_club,
        },
        checkpoint_path,
    )

    print("Best Val Loss:", best_val_loss)
    print("Saved best model:", best_model_path)
    print("Saved best CLUB:", best_club_path)
    print("Saved checkpoint:", checkpoint_path)

    # Test evaluation.
    model.load_state_dict(best_model)
    club.load_state_dict(best_club)
    model.to(device)
    club.to(device)

    _ = evaluate_test_with_club(
        model=model,
        club=club,
        test_dataset=test_dataset,
        device=device,
        batch_size=64,
        lambda_club=lambda_club,
        club_sampled_test=False,
        results_dir=results_dir,
    )
