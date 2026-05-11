import os
import numpy as np
import matplotlib.pyplot as plt

def visualize(mri, pet, recon_mri, recon_pet, save_path=None):

    mri = mri.detach().cpu().numpy().flatten()
    pet = pet.detach().cpu().numpy().flatten()
    rec_mri = recon_mri.detach().cpu().numpy().flatten()
    rec_pet = recon_pet.detach().cpu().numpy().flatten()

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    # MRI
    axes[0, 0].plot(mri, color="green")
    axes[0, 0].set_title("MRI GT")

    axes[0, 1].plot(rec_mri, color="blue")
    axes[0, 1].set_title("MRI Recon")

    # PET
    axes[1, 0].plot(pet, color="orange")
    axes[1, 0].set_title("PET GT")

    axes[1, 1].plot(rec_pet, color="red")
    axes[1, 1].set_title("PET Recon")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.close()