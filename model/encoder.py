"""Encoder: maps observations into compact latent space.

Two encoder types:
  - MLPEncoder: for structured vector observations (Snake grid, state-based)
  - ViTEncoder: for pixel observations (Push-T, Atari, Minecraft)
    Uses HuggingFace ViT-Tiny (same as LeWM paper): 5.5M params,
    patch_size=14, 12 layers, 3 heads, hidden_dim=192.

Both follow LeWM's design: encoder → CLS/hidden → BatchNorm projector → z
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """MLP encoder for structured vector observations (Snake, commodities).

    Architecture: obs → MLP → hidden → BatchNorm projector → latent z
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


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for pixel observations.

    Uses HuggingFace ViT (same architecture as LeWM paper).
    Extracts CLS token → projects through BatchNorm projector.

    Default: ViT-Tiny (5.5M params, hidden=192, 12 layers, 3 heads, patch=14)
    """

    def __init__(self, embed_dim: int = 192, image_size: int = 224,
                 patch_size: int = 14, hidden_size: int = 192,
                 num_layers: int = 12, num_heads: int = 3,
                 pretrained: bool = False):
        super().__init__()
        self.embed_dim = embed_dim

        from transformers import ViTModel, ViTConfig

        if pretrained:
            self.vit = ViTModel.from_pretrained("facebook/dino-vits14")
        else:
            config = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                num_channels=3,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=hidden_size * 4,
            )
            self.vit = ViTModel(config)

        vit_hidden = self.vit.config.hidden_size

        # Projector with BatchNorm (critical for SIGReg — see LeWM paper)
        self.projector = nn.Sequential(
            nn.Linear(vit_hidden, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, pixels):
        """
        Args:
            pixels: (B, C, H, W) or (B, T, C, H, W) pixel observations
        Returns:
            z: (B, embed_dim) or (B, T, embed_dim) latent embeddings
        """
        if pixels.ndim == 5:
            B, T, C, H, W = pixels.shape
            flat = pixels.reshape(B * T, C, H, W)
            out = self.vit(flat, interpolate_pos_encoding=True)
            cls_token = out.last_hidden_state[:, 0]
            z = self.projector(cls_token)
            return z.reshape(B, T, -1)
        else:
            out = self.vit(pixels, interpolate_pos_encoding=True)
            cls_token = out.last_hidden_state[:, 0]
            return self.projector(cls_token)
