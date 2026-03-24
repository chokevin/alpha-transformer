"""Predictor: models dynamics in latent space.

Given current latent state z_t and action a_t, predicts next latent z_{t+1}.
Uses a Transformer with Adaptive Layer Norm for action conditioning (from LeWM).

Supports both continuous actions (commodities) and discrete actions (Snake).
"""

import torch
import torch.nn as nn
import math


class ActionAdaLN(nn.Module):
    """Adaptive Layer Norm conditioned on action embeddings."""

    def __init__(self, hidden_dim: int, action_embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.scale_shift = nn.Linear(action_embed_dim, hidden_dim * 2)
        # Zero-init for stable training (from LeWM/DiT)
        nn.init.zeros_(self.scale_shift.weight)
        nn.init.zeros_(self.scale_shift.bias)

    def forward(self, x, action_emb):
        """x: (B, T, D), action_emb: (B, T, A)"""
        scale_shift = self.scale_shift(action_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class PredictorBlock(nn.Module):
    """Single transformer block with AdaLN action conditioning."""

    def __init__(self, hidden_dim: int, n_heads: int, action_embed_dim: int,
                 mlp_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.adaln1 = ActionAdaLN(hidden_dim, action_embed_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.adaln2 = ActionAdaLN(hidden_dim, action_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, action_emb, mask=None):
        h = self.adaln1(x, action_emb)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        h = self.adaln2(x, action_emb)
        x = x + self.mlp(h)
        return x


class Predictor(nn.Module):
    """Autoregressive predictor: (z_t, a_t) → ẑ_{t+1}.

    Transformer with AdaLN action conditioning, following LeWM's design.
    Supports discrete (embedding lookup) or continuous (MLP) action input.
    """

    def __init__(self, embed_dim: int = 192, action_dim: int = 4,
                 hidden_dim: int = 192, n_layers: int = 6, n_heads: int = 8,
                 mlp_dim: int = 512, dropout: float = 0.1, max_seq_len: int = 200,
                 discrete_actions: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.discrete_actions = discrete_actions

        # Action embedding
        if discrete_actions:
            self.action_embed = nn.Embedding(action_dim, hidden_dim)
        else:
            self.action_embed = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        self.blocks = nn.ModuleList([
            PredictorBlock(hidden_dim, n_heads, hidden_dim, mlp_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output projection → embed_dim with BatchNorm
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, z_seq, action_seq):
        """Predict next latent states autoregressively.

        Args:
            z_seq: (B, T, embed_dim) — sequence of latent states
            action_seq: (B, T) for discrete or (B, T, action_dim) for continuous
        Returns:
            pred_z: (B, T, embed_dim) — predicted next states
        """
        B, T, _ = z_seq.shape

        h = self.input_proj(z_seq)
        act_emb = self.action_embed(action_seq)

        positions = torch.arange(T, device=z_seq.device).unsqueeze(0)
        h = h + self.pos_embed(positions)

        mask = torch.triu(torch.ones(T, T, device=z_seq.device), diagonal=1).bool()

        for block in self.blocks:
            h = block(h, act_emb, mask=mask)

        pred = self.output_proj(h.reshape(B * T, -1))
        return pred.reshape(B, T, -1)

    def predict_step(self, z_t, action_t):
        """Single-step prediction for planning rollout.

        Args:
            z_t: (B, embed_dim) current latent
            action_t: (B,) discrete or (B, action_dim) continuous
        Returns:
            z_next: (B, embed_dim) predicted next latent
        """
        # Wrap as length-1 sequence
        z_seq = z_t.unsqueeze(1)
        if self.discrete_actions:
            action_seq = action_t.unsqueeze(1)
        else:
            action_seq = action_t.unsqueeze(1)
        pred = self.forward(z_seq, action_seq)
        return pred.squeeze(1)
