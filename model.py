import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentPartition(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.sub = nn.Linear(dim, dim)
        self.m = nn.Linear(dim, dim)
        self.p = nn.Linear(dim, dim)

    def forward(self, z):
        return self.sub(z), self.m(z), self.p(z)

class Encoder(nn.Module):
    def __init__(self, in_dim=324, hidden=512, latent=128):
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
    def __init__(self, latent=128):
        super().__init__()

        self.encoder = Encoder(324, 512, latent)
        self.partition = LatentPartition(latent)

        self.decoder_m = Decoder(latent, hidden = 512, out_dim = 162)
        self.decoder_p = Decoder(latent, hidden = 512, out_dim = 162)

        self.cls_m = nn.Linear(latent, 1)
        self.cls_p = nn.Linear(latent, 1)

    @staticmethod
    def modality_mask(x_m, x_p, p_mri=0.33, p_pet=0.33):
        B = x_m.size(0)
        device = x_m.device

        rand = torch.rand(B, device=device)

        mask_m = torch.ones(B, 1, device=device)
        mask_p = torch.ones(B, 1, device=device)

        for i in range(B):
            r = rand[i].item()

            if r < p_mri:
                mask_p[i] = 0.0   # MRI only
            elif r < p_mri + p_pet:
                mask_m[i] = 0.0   # PET only

        return x_m * mask_m, x_p * mask_p, mask_m, mask_p

    def forward(self, x_m, x_p):
        device = next(self.parameters()).device
        x_m, x_p = x_m.to(device), x_p.to(device)

        # stochastic modality masking
        x_m, x_p, mask_m, mask_p = self.modality_mask(x_m, x_p)

        x = torch.cat([x_m, x_p], dim=-1)

        latent = self.encoder(x)

        # latent partition
        z_sub, z_m, z_p = self.partition(latent)

        # modality-specific classification loss
        cls_m_pred = self.cls_m(z_m)
        cls_p_pred = self.cls_p(z_p)

        loss_cls = (
            F.binary_cross_entropy_with_logits(cls_m_pred, mask_m) +
            F.binary_cross_entropy_with_logits(cls_p_pred, mask_p)
        )

        # reconstruction (modality-specific decoders)
        pred_m = self.decoder_m(z_sub + z_m)
        pred_p = self.decoder_p(z_sub + z_p)
        loss_rec = (F.mse_loss(pred_m, x_m) + F.mse_loss(pred_p, x_p)) / 2

        # disentanglement (surrogate for MI minimization) I(z_sub ; z_m), I(z_sub ; z_p)
        loss_dis = ((z_sub * z_m).pow(2).mean() + (z_sub * z_p).pow(2).mean())

        # information bottleneck (proxy form) max I(z_sub; x) - beta I(z_sub; l_c)
        # practical surrogate:
        loss_ib = z_sub.pow(2).mean()

        loss = loss_rec + 0.1 * loss_cls + 0.1 * loss_dis + 0.01 * loss_ib

        return {
            "loss": loss,
            "loss_rec": loss_rec,
            "loss_cls": loss_cls,
            "loss_dis": loss_dis,
            "loss_ib": loss_ib,
            "z_sub": z_sub,
            "z_m": z_m,
            "z_p": z_p,
            "latent": latent,
            "reconstruction_m": pred_m,
            "reconstruction_p": pred_p,
        }


# Test
if __name__ == "__main__":
    B, N_roi = 2, 258
    inputs = torch.rand(B, N_roi)
    inputs_m = inputs.clone()
    inputs_p = inputs.clone()
    model = AE()
    outputs = model(inputs_m, inputs_p)
    print("Loss:", outputs["loss"].item())
    print("Reconstruction shape:", outputs["z_m"].shape)
    print("Classification shape:", outputs["z_sub"].shape)
    print("Latent shape:", outputs["latent"].shape)