"""SIGReg: Sketched Isotropic Gaussian Regularizer.

Prevents representation collapse by enforcing that latent embeddings
are approximately Gaussian-distributed. From LeWM/LeJEPA.

This is the key insight: instead of complex tricks (EMA, stop-gradient),
just regularize the latent distribution to be Gaussian.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Regularizer that encourages latent embeddings to follow N(0, I).

    Projects embeddings onto random directions and tests for normality
    using the Kolmogorov-Smirnov-like statistic.

    Args:
        embed_dim: dimension of latent embeddings
        num_proj: number of random projections (more = better estimate, slower)
        knots: number of quantile points for normality test
    """

    def __init__(self, embed_dim: int = 192, num_proj: int = 512, knots: int = 17):
        super().__init__()
        self.num_proj = num_proj
        self.knots = knots

        # Fixed random projection directions (not learned)
        proj = torch.randn(embed_dim, num_proj)
        proj = proj / proj.norm(dim=0, keepdim=True)
        self.register_buffer("proj_directions", proj)

        # Standard normal quantiles for KS-like test
        quantiles = torch.linspace(0.01, 0.99, knots)
        normal_cdf = torch.erfinv(2 * quantiles - 1) * (2 ** 0.5)
        self.register_buffer("target_quantiles", normal_cdf)

    def forward(self, z):
        """
        Args:
            z: (T, B, D) or (B, D) — latent embeddings
        Returns:
            loss: scalar — how far the distribution is from Gaussian
        """
        if z.ndim == 3:
            z = z.reshape(-1, z.shape[-1])  # flatten to (N, D)

        # Standardize
        z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-8)

        # Project onto random directions: (N, D) × (D, P) → (N, P)
        projected = z @ self.proj_directions  # (N, num_proj)

        # Compute empirical quantiles along each projection
        N = projected.shape[0]
        sorted_proj, _ = torch.sort(projected, dim=0)

        # Sample at quantile positions
        indices = (self.target_quantiles * 0.5 + 0.5) * (N - 1)
        indices = indices.long().clamp(0, N - 1)
        empirical_quantiles = sorted_proj[indices]  # (knots, num_proj)

        # Compare to target (standard normal quantiles)
        target = self.target_quantiles.unsqueeze(1).expand_as(empirical_quantiles)
        loss = (empirical_quantiles - target).pow(2).mean()

        return loss
