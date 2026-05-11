import os
import pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = "./results"
SMOOTH_WINDOW = 1


def running_average(x, window=5):
    """
    Centered-ish running average using convolution.
    Keeps the output length the same as input.
    """
    x = np.asarray(x, dtype=float)

    if window <= 1:
        return x

    if len(x) < window:
        return x

    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def plot_curves(history, keys, labels, title, ylabel, save_path, smooth_window=5):
    plt.figure(figsize=(8.5, 4.8))

    for key, label in zip(keys, labels):
        if key not in history:
            print(f"Warning: {key} not found in history. Skipping.")
            continue

        y = running_average(history[key], window=smooth_window)
        plt.plot(y, label=label, linewidth=2)

    plt.xlabel("Epoch", fontsize = 14, fontweight = "bold")
    plt.ylabel(ylabel, fontsize = 14, fontweight = "bold")
    plt.title(title, fontweight="bold")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")


with open(os.path.join(RESULTS_DIR, "training_history.pkl"), "rb") as f:
    h = pickle.load(f)


# ------------------------------------------------------------
# Plot 1: Main training dynamics
# ------------------------------------------------------------
plot_curves(
    history=h,
    keys=[
        "train_objective_losses",
        "train_ae_losses",
        "rec_mri_losses",
        "rec_pet_losses",
        "val_losses",
    ],
    labels=[
        "Train objective",
        "Train AE loss",
        "MRI reconstruction",
        "PET reconstruction",
        "Validation AE loss",
    ],
    title="Training Dynamics",
    ylabel="Loss",
    save_path=os.path.join(RESULTS_DIR, "presentation_training_losses.png"),
    smooth_window=SMOOTH_WINDOW,
)


# ------------------------------------------------------------
# Plot 2: Disentanglement behavior
# ------------------------------------------------------------
plot_curves(
    history=h,
    keys=[
        "train_zs_accs",
        "val_zs_accs",
        "train_zm_accs",
        "val_zm_accs",
    ],
    labels=[
        r"Train acc using $z_s$",
        r"Val acc using $z_s$",
        r"Train acc using $z_r$",
        r"Val acc using $z_r$",
    ],
    title="Disease Predictability from Latent Spaces",
    ylabel="Diagnosis accuracy",
    save_path=os.path.join(RESULTS_DIR, "presentation_disentanglement_accuracy.png"),
    smooth_window=SMOOTH_WINDOW,
)


# ------------------------------------------------------------
# Optional Plot 3: MI/adversarial diagnostics
# ------------------------------------------------------------
plot_curves(
    history=h,
    keys=[
        "mi_penalties",
        "mi_estimates",
        "probe_losses",
        "mod_entropies",
    ],
    labels=[
        "ReLU CLUB penalty",
        "Raw CLUB estimate",
        r"$z_r$ probe CE loss",
        r"Probe entropy on $z_r$",
    ],
    title="Disentanglement Regularization Diagnostics",
    ylabel="Loss / estimate",
    save_path=os.path.join(RESULTS_DIR, "presentation_regularization_diagnostics.png"),
    smooth_window=SMOOTH_WINDOW,
)

# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import os

# RESULTS_DIR = "./results"

# with open(os.path.join(RESULTS_DIR, "training_history.pkl"), "rb") as f:
#     h = pickle.load(f)

# # --- Loss curve ---
# plt.figure(figsize=(9, 5))
# plt.plot(h["train_objective_losses"], label="Train objective")
# plt.plot(h["train_ae_losses"],        label="Train AE loss")
# plt.plot(h["rec_mri_losses"],         label="Reconstruction MRI")
# plt.plot(h["rec_pet_losses"],         label="Reconstruction PET")
# plt.plot(h["rec_losses"],             label="Reconstruction loss")
# plt.plot(h["cls_losses"],             label="Classification loss")
# plt.plot(h["val_losses"],             label="Validation AE loss")
# plt.plot(h["mi_losses"],              label="CLUB learning loss")
# plt.plot(h["mi_estimates"],           label="Raw CLUB estimate")
# plt.plot(h["mi_penalties"],           label="ReLU CLUB penalty")
# plt.plot(h["adv_losses"],             label="Adv loss: -H(y|z_m)")
# plt.plot(h["probe_losses"],           label="z_m probe CE loss")
# plt.plot(h["mod_entropies"],          label="Entropy of probe on z_m")
# plt.xlabel("epoch")
# plt.ylabel("loss / estimate")
# plt.legend(fontsize=8)
# plt.title("Training Curve")
# plt.tight_layout()
# plt.savefig(os.path.join(RESULTS_DIR, "loss_curve_replot.png"), dpi=150)
# plt.close()
# print("Saved loss curve.")

# # --- Disentanglement curve ---
# plt.figure(figsize=(9, 5))
# plt.plot(h["train_zs_accs"], label="Train acc using z_s")
# plt.plot(h["train_zm_accs"], label="Train acc using z_m")
# plt.plot(h["val_zs_accs"],   label="Val acc using z_s")
# plt.plot(h["val_zm_accs"],   label="Val acc using z_m")
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.legend(fontsize=8)
# plt.title("Disentanglement Curve")
# plt.tight_layout()
# plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve_replot.png"), dpi=150)
# plt.close()
# print("Saved accuracy curve.")