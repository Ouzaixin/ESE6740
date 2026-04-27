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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_running_sums(running, out, batch_size):
    """
    Accumulate loss components over a full epoch.
    """
    keys = ["loss", "loss_rec", "loss_cls", "loss_dis", "loss_kl"]

    for key in keys:
        if key in out:
            running[key] += out[key].item() * batch_size


def normalize_running_sums(running, n_samples):
    """
    Convert accumulated sums into dataset averages.
    """
    return {key: value / n_samples for key, value in running.items()}


def train_model(
    model,
    train_dataset,
    val_dataset,
    device,
    epochs=100,
    lr=1e-4,
    batch_size=64,
    plot_path="loss_curve.png",
    results_dir="results",
):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model = None

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

        train_running = {
            "loss": 0.0,
            "loss_rec": 0.0,
            "loss_cls": 0.0,
            "loss_dis": 0.0,
            "loss_kl": 0.0,
        }

        for mri, pet in train_loader:
            mri = mri.to(device)
            pet = pet.to(device)

            optimizer.zero_grad()

            # During training, model.forward(..., apply_mask=None)
            # applies stochastic modality masking by default.
            out = model(mri, pet)

            loss = out["loss"]
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = mri.size(0)
            update_running_sums(train_running, out, batch_n)

        train_metrics = normalize_running_sums(train_running, len(train_dataset))
        train_losses.append(train_metrics["loss"])

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        val_running = {
            "loss": 0.0,
            "loss_rec": 0.0,
            "loss_cls": 0.0,
            "loss_dis": 0.0,
            "loss_kl": 0.0,
        }

        last_val_batch = None

        with torch.no_grad():
            for mri, pet in val_loader:
                mri = mri.to(device)
                pet = pet.to(device)

                # In eval mode, the updated model defaults to no masking.
                # This gives stable validation using both modalities.
                out = model(mri, pet)

                batch_n = mri.size(0)
                update_running_sums(val_running, out, batch_n)

                last_val_batch = (mri, pet, out)

        val_metrics = normalize_running_sums(val_running, len(val_dataset))
        val_losses.append(val_metrics["loss"])

        # -------------------------
        # Save best model
        # -------------------------
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model = copy.deepcopy(model.state_dict())

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
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Training Curve")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(
            f"Epoch {ep + 1:03d}/{epochs} | "
            f"Train Loss {train_metrics['loss']:.4f} "
            f"(Rec {train_metrics['loss_rec']:.4f}, "
            f"Cls {train_metrics['loss_cls']:.4f}, "
            f"Dis {train_metrics['loss_dis']:.4f}, "
            f"KL {train_metrics['loss_kl']:.4f}) | "
            f"Val Loss {val_metrics['loss']:.4f} "
            f"(Rec {val_metrics['loss_rec']:.4f}, "
            f"Cls {val_metrics['loss_cls']:.4f}, "
            f"Dis {val_metrics['loss_dis']:.4f}, "
            f"KL {val_metrics['loss_kl']:.4f})"
        )

    return best_val_loss, best_model


def evaluate_test(model, test_dataset, device, batch_size=64, results_dir="results"):
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.eval()

    test_running = {
        "loss": 0.0,
        "loss_rec": 0.0,
        "loss_cls": 0.0,
        "loss_dis": 0.0,
        "loss_kl": 0.0,
    }

    total_n = 0

    with torch.no_grad():
        for idx, (mri, pet) in enumerate(test_loader):
            mri = mri.to(device)
            pet = pet.to(device)

            out = model(mri, pet)

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

            batch_n = mri.size(0)
            update_running_sums(test_running, out, batch_n)
            total_n += batch_n

    test_metrics = normalize_running_sums(test_running, total_n)

    print(
        f"Test Loss {test_metrics['loss']:.4f} "
        f"(Rec {test_metrics['loss_rec']:.4f}, "
        f"Cls {test_metrics['loss_cls']:.4f}, "
        f"Dis {test_metrics['loss_dis']:.4f}, "
        f"KL {test_metrics['loss_kl']:.4f})"
    )

    return test_metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    curr_dir = os.getcwd()
    results_dir = os.path.join(curr_dir, "results")
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

    # Model training.
    model = AE(
        n_rois=n_rois,
        latent=128,
        hidden=512,
        lambda_cls=0.1,
        lambda_dis=0.1,
        beta_kl=0.01,
    )

    best_val_loss, best_model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        epochs=200,
        lr=2e-4,
        batch_size=64,
        plot_path=os.path.join(results_dir, "loss_curve.png"),
        results_dir=results_dir,
    )

    best_path = os.path.join(results_dir, "best.pth")
    torch.save(best_model, best_path)

    print("Best Val Loss:", best_val_loss)
    print("Saved best model:", best_path)

    # Test evaluation.
    model.load_state_dict(best_model)
    model.to(device)

    test_metrics = evaluate_test(
        model=model,
        test_dataset=test_dataset,
        device=device,
        batch_size=64,
        results_dir=results_dir,
    )