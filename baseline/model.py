import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        positive = - (mu - y_samples)**2 /2./logvar.exp() - 0.5 * logvar 
        
        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)

        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() - 0.5 * logvar

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class ROIPositionalEncoding(nn.Module):
    def __init__(self, num_rois):
        super().__init__()

        pe = torch.zeros(num_rois)
        position = torch.arange(0, num_rois, dtype=torch.float)  
        div_term = torch.exp(torch.arange(0, num_rois, 2).float() * (-math.log(10000.0) / num_rois))
        
        pe[0::2] = torch.sin(position[0::2] * div_term[:len(position[0::2])])
        pe[1::2] = torch.cos(position[1::2] * div_term[:len(position[1::2])])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe

class LatentPartition(nn.Module):
    def __init__(self, dim, subtype_dim=4):
        super().__init__()
        self.sub = nn.Linear(dim, subtype_dim)
        self.mod = nn.Linear(dim, dim)

    def forward(self, z):
        return self.sub(z), self.mod(z)

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden, latent, dropout_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, latent),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent, hidden, out_dim, dropout_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, out_dim)
        )
        
    def forward(self, z):
        return self.net(z)

class AE(nn.Module):
    def __init__(self, num_rois, latent, dim, subtype_dim, beta_mi, beta_ib, dropout_p, num_classes):
        super().__init__()
        self.encoder = Encoder(num_rois * 2, dim, latent, dropout_p=dropout_p)
        # self.partition = LatentPartition(latent)
        self.partition = LatentPartition(latent, subtype_dim=subtype_dim)
        self.decoder_mod_m = Decoder(latent, dim, num_rois, dropout_p=dropout_p)
        self.decoder_mod_p = Decoder(latent, dim, num_rois, dropout_p=dropout_p)
        self.decoder_sub_m = Decoder(subtype_dim, dim, num_rois, dropout_p=dropout_p)
        self.decoder_sub_p = Decoder(subtype_dim, dim, num_rois, dropout_p=dropout_p)
        
        self.cls_apoe = nn.Sequential(
            nn.Linear(subtype_dim, subtype_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(subtype_dim, 3)
        )

        self.reg_age = nn.Sequential(
            nn.Linear(subtype_dim, subtype_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(subtype_dim, 1)
        )

        self.cls_sub = nn.Sequential(
            nn.Linear(subtype_dim, subtype_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(subtype_dim, num_classes)
        )
        self.pos_emb = ROIPositionalEncoding(num_rois=num_rois)
        
        self.beta_mi = beta_mi
        self.beta_ib = beta_ib
        self.register_buffer("prior_mean", torch.zeros(latent))
        self.register_buffer("prior_std", torch.ones(latent))

    def forward(self, x_m, x_p, y):
        device = next(self.parameters()).device
        x_m, x_p = x_m.to(device), x_p.to(device)

        x_m_emb = self.pos_emb(x_m)
        x_p_emb = self.pos_emb(x_p)
        x = torch.cat([x_m_emb, x_p_emb], dim=-1)
        
        latent = self.encoder(x)
        z_sub, z_mod = self.partition(latent)

        mod_pred_m = self.decoder_mod_m(z_mod)
        mod_pred_p = self.decoder_mod_p(z_mod)
        
        sub_pred_m = self.decoder_sub_m(z_sub)
        sub_pred_p = self.decoder_sub_p(z_sub)

        pred_m = mod_pred_m + sub_pred_m
        pred_p = mod_pred_p + sub_pred_p
        
        # loss_rec = (F.mse_loss(pred_m, x_m) + F.mse_loss(pred_p, x_p)) / 2
        loss_rec_mri = F.mse_loss(pred_m, x_m)
        loss_rec_pet = F.mse_loss(pred_p, x_p)
        # loss_rec = (loss_rec_mri + loss_rec_pet) / 2
        loss_rec = loss_rec_mri + loss_rec_pet
        cls_pred = self.cls_sub(z_sub)
        loss_cls = F.cross_entropy(cls_pred, y.long(), label_smoothing=0.1)
        apoe_pred = self.cls_apoe(z_sub)
        age_pred = self.reg_age(z_sub).squeeze(-1)
        loss = loss_rec + self.beta_ib * loss_cls

        return {
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_rec": loss_rec,
            "loss_rec_mri": loss_rec_mri,
            "loss_rec_pet": loss_rec_pet,
            "z_sub": z_sub,
            "z_mod": z_mod,
            "latent": latent,
            "reconstruction_m": pred_m,
            "reconstruction_p": pred_p,
            "apoe_pred": apoe_pred,
            "age_pred": age_pred,
        }

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
#     """
#     CLUB estimator for I(X,Y).

#     forward():      estimates MI using positive and negative pairs
#     loglikeli():    log-likelihood of q(Y|X)
#     learning_loss(): negative log-likelihood loss for training q(Y|X)
#     """
#     def __init__(self, x_dim, y_dim, hidden_size):
#         super(CLUB, self).__init__()

#         self.p_mu = nn.Sequential(
#             nn.Linear(x_dim, hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, y_dim),
#         )

#         self.p_logvar = nn.Sequential(
#             nn.Linear(x_dim, hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, y_dim),
#             nn.Tanh(),
#         )

#     def get_mu_logvar(self, x_samples):
#         mu = self.p_mu(x_samples)
#         logvar = self.p_logvar(x_samples)
#         return mu, logvar

#     def forward(self, x_samples, y_samples):
#         mu, logvar = self.get_mu_logvar(x_samples)

#         positive = - (mu - y_samples) ** 2 / 2.0 / logvar.exp() - 0.5 * logvar

#         prediction_1 = mu.unsqueeze(1)
#         y_samples_1 = y_samples.unsqueeze(0)

#         negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.0 / logvar.exp() - 0.5 * logvar

#         return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

#     def loglikeli(self, x_samples, y_samples):
#         mu, logvar = self.get_mu_logvar(x_samples)
#         return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

#     def learning_loss(self, x_samples, y_samples):
#         return -self.loglikeli(x_samples, y_samples)


# class ROIPositionalEncoding(nn.Module):
#     """
#     Old scalar ROI positional encoding.

#     Kept for backward compatibility, but not used by the new attention encoder.
#     The attention encoder below uses vector-valued sinusoidal positional encodings.
#     """
#     def __init__(self, num_rois):
#         super().__init__()

#         pe = torch.zeros(num_rois)
#         position = torch.arange(0, num_rois, dtype=torch.float)
#         div_term = torch.exp(torch.arange(0, num_rois, 2).float() * (-math.log(10000.0) / num_rois))

#         pe[0::2] = torch.sin(position[0::2] * div_term[:len(position[0::2])])
#         pe[1::2] = torch.cos(position[1::2] * div_term[:len(position[1::2])])

#         self.register_buffer("pe", pe.unsqueeze(0))

#     def forward(self, x):
#         return x + self.pe


# class LatentPartition(nn.Module):
#     def __init__(self, dim, subtype_dim=4):
#         super().__init__()
#         self.sub = nn.Linear(dim, subtype_dim)
#         self.mod = nn.Linear(dim, dim)

#     def forward(self, z):
#         return self.sub(z), self.mod(z)


# # ---------------------------------------------------------------------
# # OLD MLP ENCODER
# # ---------------------------------------------------------------------
# # If the attention encoder is unstable, comment out the AttentionEncoder
# # below and uncomment this class. Then set:
# #
# #     Encoder = MLPEncoder
# #
# # before the AE class.
# # ---------------------------------------------------------------------
# #
# # class MLPEncoder(nn.Module):
# #     def __init__(self, in_dim, hidden, latent, dropout_p):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(in_dim, hidden),
# #             nn.ReLU(),
# #             nn.Dropout(dropout_p),
# #             nn.Linear(hidden, latent),
# #             nn.ReLU(),
# #             nn.Dropout(dropout_p),
# #         )
# #
# #     def forward(self, x):
# #         return self.net(x)


# def build_sinusoidal_positional_encoding(seq_len, d_model):
#     """
#     Standard sinusoidal positional encoding.

#     Returns:
#         pe: [1, seq_len, d_model]
#     """
#     pe = torch.zeros(seq_len, d_model)
#     position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

#     div_term = torch.exp(
#         torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
#     )

#     pe[:, 0::2] = torch.sin(position * div_term)

#     if d_model % 2 == 0:
#         pe[:, 1::2] = torch.cos(position * div_term)
#     else:
#         pe[:, 1::2] = torch.cos(position * div_term[:-1])

#     return pe.unsqueeze(0)


# class AttentionEncoder(nn.Module):
#     """
#     Attention-based encoder for concatenated MRI/PET ROI vectors.

#     Input:
#         x: [B, 2 * num_rois]

#     Internally:
#         - split into 2*num_rois scalar ROI tokens
#         - project each scalar ROI value to attn_dim
#         - add ROI positional encoding
#         - add modality embedding: MRI vs PET
#         - prepend a CLS token
#         - process using Transformer encoder
#         - map CLS output to latent dimension
#     """
#     def __init__(
#         self,
#         in_dim,
#         hidden,
#         latent,
#         dropout_p,
#         attn_dim=64,
#         num_heads=4,
#         num_layers=2,
#     ):
#         super().__init__()

#         assert in_dim % 2 == 0, "Expected input dimension to be 2 * num_rois."

#         self.in_dim = in_dim
#         self.num_rois = in_dim // 2
#         self.seq_len = in_dim
#         self.attn_dim = attn_dim

#         # Each ROI value is scalar, projected to token embedding.
#         self.value_proj = nn.Linear(1, attn_dim)

#         # Modality embedding: 0 = MRI, 1 = PET.
#         self.modality_emb = nn.Embedding(2, attn_dim)

#         modality_ids = torch.cat([
#             torch.zeros(self.num_rois, dtype=torch.long),
#             torch.ones(self.num_rois, dtype=torch.long),
#         ])
#         self.register_buffer("modality_ids", modality_ids)

#         # Sinusoidal positional encoding over the 2*num_rois token sequence.
#         pe = build_sinusoidal_positional_encoding(self.seq_len, attn_dim)
#         self.register_buffer("pos_encoding", pe)

#         # CLS token summarizes the full multimodal sequence.
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, attn_dim))

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=attn_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden,
#             dropout=dropout_p,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,
#         )

#         self.transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers,
#         )

#         self.out = nn.Sequential(
#             nn.LayerNorm(attn_dim),
#             nn.Linear(attn_dim, latent),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#         )

#         self._reset_parameters()

#     def _reset_parameters(self):
#         nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

#     def forward(self, x):
#         """
#         x: [B, 2*num_rois]
#         """
#         B = x.size(0)

#         # [B, 2*num_rois] -> [B, 2*num_rois, 1]
#         tokens = x.unsqueeze(-1)

#         # [B, 2*num_rois, 1] -> [B, 2*num_rois, attn_dim]
#         tokens = self.value_proj(tokens)

#         # Add positional encoding.
#         tokens = tokens + self.pos_encoding[:, :self.seq_len, :]

#         # Add modality embedding.
#         modality_embed = self.modality_emb(self.modality_ids).unsqueeze(0)
#         tokens = tokens + modality_embed

#         # Add CLS token.
#         cls = self.cls_token.expand(B, -1, -1)
#         tokens = torch.cat([cls, tokens], dim=1)

#         # Transformer encoder.
#         encoded = self.transformer(tokens)

#         # CLS output.
#         cls_out = encoded[:, 0, :]

#         return self.out(cls_out)


# # Use attention encoder by default.
# Encoder = AttentionEncoder

# # To revert to the old MLP encoder:
# #   1. Uncomment MLPEncoder above.
# #   2. Replace the line above with:
# #      Encoder = MLPEncoder


# class Decoder(nn.Module):
#     def __init__(self, latent, hidden, out_dim, dropout_p):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(latent, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(hidden, out_dim),
#         )

#     def forward(self, z):
#         return self.net(z)


# class AE(nn.Module):
#     def __init__(
#         self,
#         num_rois,
#         latent,
#         dim,
#         subtype_dim,
#         beta_mi,
#         beta_ib,
#         dropout_p,
#         num_classes,
#     ):
#         super().__init__()

#         self.num_rois = num_rois
#         self.latent = latent
#         self.subtype_dim = subtype_dim

#         self.encoder = Encoder(
#             num_rois * 2,
#             dim,
#             latent,
#             dropout_p=dropout_p,
#         )

#         self.partition = LatentPartition(
#             latent,
#             subtype_dim=subtype_dim,
#         )

#         self.decoder_mod_m = Decoder(
#             latent,
#             dim,
#             num_rois,
#             dropout_p=dropout_p,
#         )

#         self.decoder_mod_p = Decoder(
#             latent,
#             dim,
#             num_rois,
#             dropout_p=dropout_p,
#         )

#         self.decoder_sub_m = Decoder(
#             subtype_dim,
#             dim,
#             num_rois,
#             dropout_p=dropout_p,
#         )

#         self.decoder_sub_p = Decoder(
#             subtype_dim,
#             dim,
#             num_rois,
#             dropout_p=dropout_p,
#         )

#         self.cls_apoe = nn.Sequential(
#             nn.Linear(subtype_dim, subtype_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(subtype_dim, 3),
#         )

#         self.reg_age = nn.Sequential(
#             nn.Linear(subtype_dim, subtype_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(subtype_dim, 1),
#         )

#         self.cls_sub = nn.Sequential(
#             nn.Linear(subtype_dim, subtype_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(subtype_dim, num_classes),
#         )

#         # Kept for backward compatibility with old MLP encoder.
#         # Not used when AttentionEncoder is active.
#         self.pos_emb = ROIPositionalEncoding(num_rois=num_rois)

#         self.beta_mi = beta_mi
#         self.beta_ib = beta_ib

#         self.register_buffer("prior_mean", torch.zeros(latent))
#         self.register_buffer("prior_std", torch.ones(latent))

#     def forward(self, x_m, x_p, y):
#         device = next(self.parameters()).device
#         x_m, x_p = x_m.to(device), x_p.to(device)

#         # -------------------------------------------------------------
#         # OLD MLP-style positional encoding.
#         # Commented out because the AttentionEncoder now adds
#         # vector-valued ROI positional encodings internally.
#         # -------------------------------------------------------------
#         # x_m_emb = self.pos_emb(x_m)
#         # x_p_emb = self.pos_emb(x_p)
#         # x = torch.cat([x_m_emb, x_p_emb], dim=-1)

#         # AttentionEncoder expects raw concatenated MRI/PET vectors.
#         x = torch.cat([x_m, x_p], dim=-1)

#         latent = self.encoder(x)
#         z_sub, z_mod = self.partition(latent)

#         mod_pred_m = self.decoder_mod_m(z_mod)
#         mod_pred_p = self.decoder_mod_p(z_mod)

#         sub_pred_m = self.decoder_sub_m(z_sub)
#         sub_pred_p = self.decoder_sub_p(z_sub)

#         pred_m = mod_pred_m + sub_pred_m
#         pred_p = mod_pred_p + sub_pred_p

#         loss_rec_mri = F.mse_loss(pred_m, x_m)
#         loss_rec_pet = F.mse_loss(pred_p, x_p)

#         # Current implementation uses sum, not average.
#         loss_rec = loss_rec_mri + loss_rec_pet

#         cls_pred = self.cls_sub(z_sub)

#         loss_cls = F.cross_entropy(
#             cls_pred,
#             y.long(),
#             label_smoothing=0.1,
#         )

#         apoe_pred = self.cls_apoe(z_sub)
#         age_pred = self.reg_age(z_sub).squeeze(-1)

#         loss = loss_rec + self.beta_ib * loss_cls

#         return {
#             "loss": loss,
#             "loss_cls": loss_cls,
#             "loss_rec": loss_rec,
#             "loss_rec_mri": loss_rec_mri,
#             "loss_rec_pet": loss_rec_pet,
#             "z_sub": z_sub,
#             "z_mod": z_mod,
#             "latent": latent,
#             "reconstruction_m": pred_m,
#             "reconstruction_p": pred_p,
#             "apoe_pred": apoe_pred,
#             "age_pred": age_pred,
#         }

