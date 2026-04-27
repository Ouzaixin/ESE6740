# club_gaussian_experiment.py

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_correlated_gaussian(batch_size, dim, rho, device):
    """
    X \sim N(0, I)
    Y = rho X + sqrt(1-rho^2) eps
    eps \sim N(0, I)

    True MI:
        I(X;Y) = - d/2 log(1-rho^2)
    """
    x = torch.randn(batch_size, dim, device=device)
    eps = torch.randn(batch_size, dim, device=device)
    y = rho * x + math.sqrt(1.0 - rho ** 2) * eps
    return x, y


def true_gaussian_mi(dim, rho):
    return -0.5 * dim * math.log(1.0 - rho ** 2)


class CLUBGaussian(nn.Module):
    """
    Variational CLUB with Gaussian q_theta(y|x):

        q_theta(y|x) = N(mu_theta(x), diag(sigma_theta^2(x)))

    Used for estimating/minimizing I(X;Y).
    """

    def __init__(self, x_dim, y_dim, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, y_dim)
        self.logvar_head = nn.Linear(hidden_dim, y_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)

        logvar = self.logvar_head(h).clamp(min=-8.0, max=5.0)
        return mu, logvar

    def log_likelihood(self, x, y):
        """
        Mean log q_theta(y|x).

        This is used to train q_theta.
        """
        mu, logvar = self.forward(x)

        log_prob = -0.5 * (
            ((y - mu) ** 2) / logvar.exp()
            + logvar
            + math.log(2.0 * math.pi)
        )

        return log_prob.sum(dim=1).mean()

    def mi_estimate_full(self, x, y):
        """
        Full vCLUB estimate:

            E_joint log q(y|x) - E_marginal log q(y|x)

        Uses all negative pairs in the batch.
        Complexity: O(B^2).
        """
        mu, logvar = self.forward(x)

        # Positive term: log q(y_i | x_i)
        positive = -0.5 * (((y - mu) ** 2) / logvar.exp()).sum(dim=1)

        # Negative term: log q(y_j | x_i), averaged over all j
        # Shapes:
        #   mu:     [B, D] -> [B, 1, D]
        #   y:      [B, D] -> [1, B, D]
        #   logvar: [B, D] -> [B, 1, D]
        mu_i = mu.unsqueeze(1)
        logvar_i = logvar.unsqueeze(1)
        y_j = y.unsqueeze(0)

        negative = -0.5 * (((y_j - mu_i) ** 2) / logvar_i.exp()).sum(dim=2)

        # positive: [B]
        # negative: [B, B]
        return positive.mean() - negative.mean()

    def mi_estimate_sampled(self, x, y):
        """
        Sampled vCLUB estimate:

            1/B sum_i [log q(y_i|x_i) - log q(y_perm_i|x_i)]

        Complexity: O(B).
        """
        batch_size = x.shape[0]
        mu, logvar = self.forward(x)
        positive = -0.5 * (((y - mu) ** 2) / logvar.exp()).sum(dim=1)

        # Randomly permute y to create negative samples.
        perm = torch.randperm(batch_size, device=x.device)

        # Avoid identity permutation as much as possible.
        if torch.any(perm == torch.arange(batch_size, device=x.device)):
            perm = torch.roll(perm, shifts=1)

        y_neg = y[perm]
        negative = -0.5 * (((y_neg - mu) ** 2) / logvar.exp()).sum(dim=1)

        return positive.mean() - negative.mean()


# Experiment 1:
# Estimate MI for fixed Gaussian data
# Given dependent Gaussian variables X,Y, can CLUB estimate their MI?
def run_club_estimation_experiment(
    dim=20,
    rho=0.7,
    batch_size=256,
    steps=5000,
    lr=1e-3,
    use_sampled=False,
    seed=0,
    device=None,
):
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    club = CLUBGaussian(x_dim=dim, y_dim=dim, hidden_dim=128).to(device)
    optimizer = optim.Adam(club.parameters(), lr=lr)

    true_mi = true_gaussian_mi(dim, rho)

    for step in range(1, steps + 1):
        x, y = sample_correlated_gaussian(batch_size, dim, rho, device)

        # Train q_theta by maximizing log q_theta(y|x).
        # Equivalently minimize negative log-likelihood.
        ll = club.log_likelihood(x, y)
        loss_q = -ll

        optimizer.zero_grad()
        loss_q.backward()
        optimizer.step()

        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                x_eval, y_eval = sample_correlated_gaussian(
                    batch_size, dim, rho, device
                )

                if use_sampled:
                    mi_hat = club.mi_estimate_sampled(x_eval, y_eval)
                else:
                    mi_hat = club.mi_estimate_full(x_eval, y_eval)

            print(
                f"step={step:05d} | "
                f"loglik={ll.item():.3f} | "
                f"MI_hat={mi_hat.item():.3f} | "
                f"true_MI={true_mi:.3f}"
            )

    return club


# Experiment 2:
# MI minimization toy experiment
class LearnableCorrelationModel(nn.Module):
    """
    A toy 'encoder-like' model.

    We start with X ~ N(0,I), and construct

        Y = alpha X + eps

    where alpha is learnable.

    Minimizing CLUB should push alpha toward 0,
    thereby reducing dependence between X and Y.
    """

    def __init__(self, dim, init_alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.dim = dim

    def forward(self, batch_size, device):
        x = torch.randn(batch_size, self.dim, device=device)
        eps = torch.randn(batch_size, self.dim, device=device)

        y = self.alpha * x + eps
        return x, y


def run_club_minimization_experiment(
    dim=20,
    batch_size=256,
    steps=5000,
    club_lr=1e-3,
    model_lr=1e-3,
    lambda_mi=1.0,
    use_sampled=True,
    seed=0,
    device=None,
):
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LearnableCorrelationModel(dim=dim, init_alpha=1.0).to(device)
    club = CLUBGaussian(x_dim=dim, y_dim=dim, hidden_dim=128).to(device)

    opt_club = optim.Adam(club.parameters(), lr=club_lr)
    opt_model = optim.Adam(model.parameters(), lr=model_lr)

    for step in range(1, steps + 1):

        # Step 1: update CLUB estimator q_theta
        x, y = model(batch_size, device)

        # Detach x,y so this step does not update the main model.
        x_detached = x.detach()
        y_detached = y.detach()

        ll = club.log_likelihood(x_detached, y_detached)
        loss_club = -ll

        opt_club.zero_grad()
        loss_club.backward()
        opt_club.step()


        # Step 2: update main model to minimize MI
        x, y = model(batch_size, device)

        # Freeze CLUB parameters during main-model update.
        for p in club.parameters():
            p.requires_grad_(False)

        if use_sampled:
            mi_hat = club.mi_estimate_sampled(x, y)
        else:
            mi_hat = club.mi_estimate_full(x, y)

        loss_model = lambda_mi * mi_hat

        opt_model.zero_grad()
        loss_model.backward()
        opt_model.step()

        for p in club.parameters():
            p.requires_grad_(True)

        if step % 500 == 0 or step == 1:
            alpha = model.alpha.item()

            # For Y = alpha X + eps, scalar correlation per coordinate is:
            # rho = alpha / sqrt(alpha^2 + 1)
            rho_eff = alpha / math.sqrt(alpha ** 2 + 1.0)
            true_mi_eff = true_gaussian_mi(dim, rho_eff)

            print(
                f"step={step:05d} | "
                f"alpha={alpha:.4f} | "
                f"rho_eff={rho_eff:.4f} | "
                f"MI_hat={mi_hat.item():.4f} | "
                f"true_MI_eff={true_mi_eff:.4f} | "
                f"club_loglik={ll.item():.4f}"
            )

    return model, club


if __name__ == "__main__":

    # print("Experiment 1: CLUB MI Estimation")

    # _ = run_club_estimation_experiment(
    #     dim=20,
    #     rho=0.7,
    #     batch_size=256,
    #     steps=5000,
    #     lr=1e-3,
    #     use_sampled=False,
    #     seed=0,
    # )

    print("Experiment 2: CLUB MI Minimization")

    _ = run_club_minimization_experiment(
        dim=20,
        batch_size=256,
        steps=5000,
        club_lr=1e-3,
        model_lr=1e-3,
        lambda_mi=1.0,
        use_sampled=True,
        seed=0,
    )