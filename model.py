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
    def __init__(self, dim):
        super().__init__()

        self.sub = nn.Linear(dim, dim//32)
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
    def __init__(self, num_rois, latent, dim, beta_mi, beta_ib, dropout_p, num_classes):
        super().__init__()
        self.encoder = Encoder(num_rois * 2, dim, latent, dropout_p=dropout_p)
        self.partition = LatentPartition(latent)

        self.decoder_mod_m = Decoder(latent, dim, num_rois, dropout_p=dropout_p)
        self.decoder_mod_p = Decoder(latent, dim, num_rois, dropout_p=dropout_p)
        self.decoder_sub_m = Decoder(latent//32, dim, num_rois, dropout_p=dropout_p)
        self.decoder_sub_p = Decoder(latent//32, dim, num_rois, dropout_p=dropout_p)

        self.cls_sub = nn.Sequential(
            nn.Linear(latent//32, dim//32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim//32, num_classes)
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
        
        loss_rec = (F.mse_loss(pred_m, x_m) + F.mse_loss(pred_p, x_p)) / 2

        cls_pred = self.cls_sub(z_sub)
        loss_cls = F.cross_entropy(cls_pred, y.long())

        loss = loss_rec + self.beta_ib * loss_cls

        return {
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_rec": loss_rec,
            "z_sub": z_sub,
            "z_mod": z_mod,
            "latent": latent,
            "reconstruction_m": pred_m,
            "reconstruction_p": pred_p,
        }