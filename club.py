# club.py

import math
import torch
import torch.nn as nn


class CLUBGaussian(nn.Module):
    """
    Variational CLUB estimator with Gaussian conditional density.

    Estimates / penalizes mutual information between two continuous
    latent variables x and y using:

        q_theta(y | x) = N(mu_theta(x), diag(sigma_theta^2(x)))

    In AD model, typical usage is:

        x = z_sub
        y = z_m

    or:

        x = z_sub
        y = z_p

    Training has two separate steps:

    1. Update CLUB parameters theta:
            minimize learning_loss(x.detach(), y.detach())

    2. Update main model parameters:
            minimize mi_estimate(x, y)

       while CLUB parameters are frozen.
    """

    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim=128,
        logvar_min=-8.0,
        logvar_max=5.0,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim

        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, y_dim)
        self.logvar_head = nn.Linear(hidden_dim, y_dim)

    def forward(self, x):
        """
        Returns:
            mu:     [B, y_dim]
            logvar: [B, y_dim]
        """
        h = self.net(x)
        mu = self.mu_head(h)

        logvar = self.logvar_head(h)
        logvar = torch.clamp(
            logvar,
            min=self.logvar_min,
            max=self.logvar_max,
        )

        return mu, logvar

    def log_prob(self, x, y, include_const=True):
        """
        Compute log q_theta(y | x).

        Args:
            x: [B, x_dim]
            y: [B, y_dim]
            include_const:
                Whether to include the Gaussian normalizing constant.

        Returns:
            log_prob: [B]
        """
        mu, logvar = self.forward(x)

        log_prob = -0.5 * (
            ((y - mu) ** 2) / logvar.exp()
            + logvar
        )

        if include_const:
            log_prob = log_prob - 0.5 * math.log(2.0 * math.pi)

        return log_prob.sum(dim=1)

    def learning_loss(self, x, y):
        """
        Negative conditional log-likelihood.

        This is used to train q_theta(y | x).

        Minimize:
            - E_joint [log q_theta(y | x)]

        Important:
            In train.py, pass detached latents:
                loss = club.learning_loss(z_sub.detach(), z_m.detach())
        """
        return -self.log_prob(x, y, include_const=True).mean()

    def loglikelihood(self, x, y):
        """
        Mean conditional log-likelihood.

        Useful for logging only.

        Higher is better for the CLUB estimator network.
        """
        return self.log_prob(x, y, include_const=True).mean()

    def mi_estimate_full(self, x, y):
        """
        Full vCLUB estimate:

            E_joint [log q(y|x)]
            -
            E_marginal [log q(y|x)]

        Minibatch estimator:

            1/B sum_i log q(y_i | x_i)
            -
            1/B^2 sum_i sum_j log q(y_j | x_i)

        Complexity:
            O(B^2)

        Args:
            x: [B, x_dim]
            y: [B, y_dim]

        Returns:
            scalar MI upper-bound-style estimate
        """
        mu, logvar = self.forward(x)

        # Positive log-probability: log q(y_i | x_i)
        positive = -0.5 * (
            ((y - mu) ** 2) / logvar.exp()
            + logvar
        ).sum(dim=1)

        # Negative log-probability: log q(y_j | x_i)
        #
        # Shapes:
        #   mu_i:     [B, 1, D]
        #   logvar_i: [B, 1, D]
        #   y_j:      [1, B, D]
        mu_i = mu.unsqueeze(1)
        logvar_i = logvar.unsqueeze(1)
        y_j = y.unsqueeze(0)

        negative = -0.5 * (
            ((y_j - mu_i) ** 2) / logvar_i.exp()
            + logvar_i
        ).sum(dim=2)

        return positive.mean() - negative.mean()

    def mi_estimate_sampled(self, x, y):
        """
        Sampled vCLUB estimate:

            1/B sum_i [
                log q(y_i | x_i)
                -
                log q(y_perm_i | x_i)
            ]

        Complexity:
            O(B)

        This is the version we likely want for the AD model once
        batch size is >= 64.

        Args:
            x: [B, x_dim]
            y: [B, y_dim]

        Returns:
            scalar sampled CLUB estimate
        """
        batch_size = x.size(0)

        if batch_size <= 1:
            raise ValueError("CLUB sampled estimate requires batch_size > 1.")

        mu, logvar = self.forward(x)

        positive = -0.5 * (
            ((y - mu) ** 2) / logvar.exp()
            + logvar
        ).sum(dim=1)

        perm = torch.randperm(batch_size, device=x.device)

        # Avoid exact self-pairs where possible.
        # Rolling by one guarantees no fixed points only if randperm is identity-like?
        # This simple correction is usually enough for minibatch training.
        same = perm == torch.arange(batch_size, device=x.device)
        if same.any():
            perm[same] = torch.roll(perm, shifts=1)[same]

        y_neg = y[perm]

        negative = -0.5 * (
            ((y_neg - mu) ** 2) / logvar.exp()
            + logvar
        ).sum(dim=1)

        return positive.mean() - negative.mean()

    def forward_mi(self, x, y, sampled=True):
        """
        Convenience wrapper.

        Args:
            sampled:
                True  -> sampled O(B) estimate
                False -> full O(B^2) estimate
        """
        if sampled:
            return self.mi_estimate_sampled(x, y)
        return self.mi_estimate_full(x, y)


class CLUBForAD(nn.Module):
    """
    Convenience wrapper for your AD latent structure.

    It contains two CLUB estimators:

        club_m: estimates I(z_sub ; z_m)
        club_p: estimates I(z_sub ; z_p)

    This wrapper is useful but optional. You could also instantiate
    two CLUBGaussian modules directly in train.py.
    """

    def __init__(
        self,
        latent_dim,
        hidden_dim=128,
        logvar_min=-8.0,
        logvar_max=5.0,
    ):
        super().__init__()

        self.club_m = CLUBGaussian(
            x_dim=latent_dim,
            y_dim=latent_dim,
            hidden_dim=hidden_dim,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )

        self.club_p = CLUBGaussian(
            x_dim=latent_dim,
            y_dim=latent_dim,
            hidden_dim=hidden_dim,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )

    def learning_loss(self, z_sub, z_m, z_p):
        """
        Loss for updating CLUB networks only.

        Use detached latents when calling this in train.py:

            loss_club = club.learning_loss(
                z_sub.detach(),
                z_m.detach(),
                z_p.detach()
            )

        Returns:
            scalar negative log-likelihood loss
        """
        loss_m = self.club_m.learning_loss(z_sub, z_m)
        loss_p = self.club_p.learning_loss(z_sub, z_p)

        return loss_m + loss_p

    def loglikelihood(self, z_sub, z_m, z_p):
        """
        Logging metric.

        Higher is better for the CLUB conditional predictors.
        """
        ll_m = self.club_m.loglikelihood(z_sub, z_m)
        ll_p = self.club_p.loglikelihood(z_sub, z_p)

        return {
            "club_ll_m": ll_m,
            "club_ll_p": ll_p,
            "club_ll_total": ll_m + ll_p,
        }

    def mi_estimate(self, z_sub, z_m, z_p, sampled=True):
        """
        CLUB MI penalty for updating the main model.

        During the main-model update, freeze CLUB parameters but do NOT
        detach z_sub, z_m, z_p. We want gradients to flow into the encoder.

        Returns:
            total_mi, mi_m, mi_p
        """
        mi_m = self.club_m.forward_mi(z_sub, z_m, sampled=sampled)
        mi_p = self.club_p.forward_mi(z_sub, z_p, sampled=sampled)

        total_mi = mi_m + mi_p

        return total_mi, mi_m, mi_p


def set_requires_grad(module, requires_grad):
    """
    Utility for freezing/unfreezing CLUB during alternating training.

    Example:

        # Update CLUB
        set_requires_grad(club, True)
        set_requires_grad(model, False)

        # Update main model
        set_requires_grad(club, False)
        set_requires_grad(model, True)
    """
    for param in module.parameters():
        param.requires_grad_(requires_grad)