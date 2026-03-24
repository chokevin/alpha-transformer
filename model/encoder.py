"""Encoder: maps observations into compact latent space.

MLP-based since our observations are structured vectors (not pixels).
Follows LeWM's design: encoder + projector with BatchNorm.

Supports both commodity (52-dim) and Snake (300-dim) observation spaces.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encode observations into latent space.

    Architecture: obs → MLP → hidden → BatchNorm projector → latent z
    The BatchNorm projector is critical for SIGReg (see LeWM paper).
    """

    def __init__(self, obs_dim: int = 300, embed_dim: int = 192, hidden_dim: int = 512):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Projector with BatchNorm (critical for SIGReg)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, obs):
        """
        Args:
            obs: (B, obs_dim) or (B, T, obs_dim)
        Returns:
            z: same leading dims, last dim = embed_dim
        """
        if obs.ndim == 3:
            B, T, D = obs.shape
            h = self.encoder(obs.reshape(B * T, D))
            z = self.projector(h)
            return z.reshape(B, T, -1)
        else:
            h = self.encoder(obs)
            return self.projector(h)
