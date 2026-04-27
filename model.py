import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPartition(nn.Module):
    """
    Partition the shared encoder latent into:
      - probabilistic subtype latent: mu_sub, logvar_sub
      - deterministic MRI-specific latent: z_m
      - deterministic PET-specific latent: z_p
    """

    def __init__(self, dim):
        super().__init__()

        self.mu_sub = nn.Linear(dim, dim)
        self.logvar_sub = nn.Linear(dim, dim)

        self.m = nn.Linear(dim, dim)
        self.p = nn.Linear(dim, dim)

    def forward(self, z):
        mu_sub = self.mu_sub(z)
        logvar_sub = self.logvar_sub(z)

        logvar_sub = torch.clamp(logvar_sub, min=-8.0, max=5.0)

        z_m = self.m(z)
        z_p = self.p(z)

        return mu_sub, logvar_sub, z_m, z_p


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden=512, latent=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent=128, hidden=512, out_dim=162):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, z):
        return self.net(z)


class AE(nn.Module):
    def __init__(
        self,
        n_rois,
        latent=128,
        hidden=512,
        lambda_cls=0.1,
        lambda_dis=0.1,
        beta_kl=0.01,
    ):
        """
        Args:
            n_rois: number of matched MRI/PET ROI pairs.
                    MRI input dim = n_rois, PET input dim = n_rois.
                    Joint input dim = 2 * n_rois.
            latent: latent dimension for shared, subtype, and modality-specific spaces.
            hidden: hidden dimension for encoder/decoders.
            lambda_cls: weight for modality-presence classification loss.
            lambda_dis: weight for current disentanglement surrogate.
            beta_kl: weight for VIB KL loss.
        """
        super().__init__()

        self.n_rois = n_rois
        self.latent_dim = latent

        self.lambda_cls = lambda_cls
        self.lambda_dis = lambda_dis
        self.beta_kl = beta_kl

        self.encoder = Encoder(in_dim=2 * n_rois, hidden=hidden, latent=latent)
        self.partition = LatentPartition(latent)

        self.decoder_m = Decoder(latent=latent, hidden=hidden, out_dim=n_rois)
        self.decoder_p = Decoder(latent=latent, hidden=hidden, out_dim=n_rois)

        self.cls_m = nn.Linear(latent, 1)
        self.cls_p = nn.Linear(latent, 1)

    @staticmethod
    def modality_mask(x_m, x_p, p_mri=0.33, p_pet=0.33):
        """
        Randomly mask modalities.

        With probability p_mri: keep MRI only, mask PET.
        With probability p_pet: keep PET only, mask MRI.
        Otherwise: keep both.
        """
        B = x_m.size(0)
        device = x_m.device

        rand = torch.rand(B, device=device)

        mask_m = torch.ones(B, 1, device=device)
        mask_p = torch.ones(B, 1, device=device)

        # Vectorized version of the original loop.
        pet_missing = rand < p_mri
        mri_missing = (rand >= p_mri) & (rand < p_mri + p_pet)

        mask_p[pet_missing] = 0.0   # MRI only
        mask_m[mri_missing] = 0.0   # PET only

        return x_m * mask_m, x_p * mask_p, mask_m, mask_p

    def reparameterize(self, mu, logvar):
        """
        VIB reparameterization.

        During training:
            z = mu + sigma * eps

        During eval:
            z = mu

        This makes inference deterministic and lets us later cluster mu_sub.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        return mu

    @staticmethod
    def kl_standard_normal(mu, logvar):
        """
        KL[q(z|x) || N(0,I)] for diagonal Gaussian q.

        q(z|x) = N(mu, diag(exp(logvar)))

        Returns average KL per subject.
        """
        kl_per_sample = -0.5 * torch.sum(
            1.0 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
        )
        return kl_per_sample.mean()

    def forward(self, x_m, x_p, apply_mask=None):
        """
        Args:
            x_m: MRI ROI tensor, shape [B, n_rois]
            x_p: PET ROI tensor, shape [B, n_rois]
            apply_mask:
                None: apply stochastic masking only during training.
                True: force stochastic masking.
                False: do not mask.

        Returns:
            dictionary containing losses, latents, reconstructions, and masks.
        """
        device = next(self.parameters()).device
        x_m = x_m.to(device)
        x_p = x_p.to(device)

        # Preserve original inputs as reconstruction targets.
        x_m_orig = x_m
        x_p_orig = x_p

        # Decide whether to use modality masking.
        if apply_mask is None:
            apply_mask = self.training

        if apply_mask:
            x_m_in, x_p_in, mask_m, mask_p = self.modality_mask(x_m_orig, x_p_orig)
        else:
            B = x_m.size(0)
            mask_m = torch.ones(B, 1, device=device)
            mask_p = torch.ones(B, 1, device=device)
            x_m_in = x_m_orig
            x_p_in = x_p_orig

        # Joint masked input.
        x = torch.cat([x_m_in, x_p_in], dim=-1)

        # Shared latent.
        latent = self.encoder(x)

        # Latent partition.
        mu_sub, logvar_sub, z_m, z_p = self.partition(latent)

        # Probabilistic subtype latent.
        z_sub = self.reparameterize(mu_sub, logvar_sub)

        # Modality-specific classification loss.
        cls_m_pred = self.cls_m(z_m)
        cls_p_pred = self.cls_p(z_p)

        loss_cls = (
            F.binary_cross_entropy_with_logits(cls_m_pred, mask_m) +
            F.binary_cross_entropy_with_logits(cls_p_pred, mask_p)
        )

        # Reconstruction.
        # Important: reconstruct original MRI/PET, not masked inputs.
        pred_m = self.decoder_m(z_sub + z_m)
        pred_p = self.decoder_p(z_sub + z_p)

        loss_rec = (
            F.mse_loss(pred_m, x_m_orig) +
            F.mse_loss(pred_p, x_p_orig)
        ) / 2.0

        # Temporary disentanglement surrogate.
        # Later, we will replace this with CLUB.
        loss_dis = (
            (z_sub * z_m).pow(2).mean() +
            (z_sub * z_p).pow(2).mean()
        )

        # True VIB KL term for z_sub.
        loss_kl = self.kl_standard_normal(mu_sub, logvar_sub)

        loss = (
            loss_rec
            + self.lambda_cls * loss_cls
            + self.lambda_dis * loss_dis
            + self.beta_kl * loss_kl
        )

        return {
            "loss": loss,
            "loss_rec": loss_rec,
            "loss_cls": loss_cls,
            "loss_dis": loss_dis,
            "loss_kl": loss_kl,

            # Backward-compatible alias for your old training logs.
            "loss_ib": loss_kl,

            "z_sub": z_sub,
            "mu_sub": mu_sub,
            "logvar_sub": logvar_sub,
            "z_m": z_m,
            "z_p": z_p,
            "latent": latent,

            "reconstruction_m": pred_m,
            "reconstruction_p": pred_p,

            "mask_m": mask_m,
            "mask_p": mask_p,
            "x_m_input": x_m_in,
            "x_p_input": x_p_in,
            "x_m_target": x_m_orig,
            "x_p_target": x_p_orig,
        }


# Test
if __name__ == "__main__":
    B, n_rois = 2, 162

    inputs_m = torch.rand(B, n_rois)
    inputs_p = torch.rand(B, n_rois)

    model = AE(n_rois=n_rois, latent=128, hidden=512)

    outputs = model(inputs_m, inputs_p)

    print("Loss:", outputs["loss"].item())
    print("MRI reconstruction shape:", outputs["reconstruction_m"].shape)
    print("PET reconstruction shape:", outputs["reconstruction_p"].shape)
    print("z_sub shape:", outputs["z_sub"].shape)
    print("mu_sub shape:", outputs["mu_sub"].shape)
    print("logvar_sub shape:", outputs["logvar_sub"].shape)
    print("z_m shape:", outputs["z_m"].shape)
    print("z_p shape:", outputs["z_p"].shape)
    print("Latent shape:", outputs["latent"].shape)