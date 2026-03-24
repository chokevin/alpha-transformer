"""Decision Transformer for multi-commodity trading.

Causal transformer that treats trading as sequence modeling:
  Input:  Interleaved tokens [return-to-go, state, action] per timestep
  Output: Predicted position sizes for each commodity

Architecture improvements over vanilla Decision Transformer:
  - Pre-norm (LayerNorm before attention) for training stability
  - Learnable timestep encoding added to positional encoding
  - Separate action prediction MLP with residual connection
  - Configurable depth/width via config
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TradingTransformer(nn.Module):
    """Causal transformer that predicts trading actions from
    sequences of (state, action, return-to-go).

    Token interleaving: [RTG_1, State_1, Action_1, RTG_2, State_2, Action_2, ...]
    Action is predicted from the state token output.
    """

    def __init__(self, state_dim=52, action_dim=4, hidden_dim=128,
                 n_heads=4, n_layers=4, max_seq_len=60, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.action_dim = action_dim

        # Token type embeddings
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.rtg_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Positional encoding (3 tokens per timestep)
        self.pos_embed = nn.Embedding(max_seq_len * 3, hidden_dim)

        # Timestep encoding (shared across token types within a step)
        self.timestep_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer encoder (pre-norm for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # actions in [-1, 1]
        )

        # Learnable log-std for exploration (RL phase)
        self.log_std = nn.Parameter(torch.full((action_dim,), 0.0))  # start with std≈1.0 for exploration

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, states, actions, rtgs, timesteps=None):
        """
        Args:
            states:  (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
            rtgs:    (batch, seq_len, 1)
            timesteps: (batch, seq_len) optional timestep indices
        Returns:
            action_preds: (batch, seq_len, action_dim)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        device = states.device

        # Embed each modality
        s_emb = self.state_embed(states)    # (B, T, H)
        a_emb = self.action_embed(actions)  # (B, T, H)
        r_emb = self.rtg_embed(rtgs)        # (B, T, H)

        # Interleave: [r1, s1, a1, r2, s2, a2, ...]
        tokens = torch.zeros(batch_size, seq_len * 3, self.hidden_dim, device=device)
        tokens[:, 0::3] = r_emb
        tokens[:, 1::3] = s_emb
        tokens[:, 2::3] = a_emb

        # Positional embeddings
        positions = torch.arange(seq_len * 3, device=device).unsqueeze(0)
        tokens = tokens + self.pos_embed(positions)

        # Timestep embeddings (same for all 3 tokens within a step)
        if timesteps is None:
            timesteps = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        timesteps = timesteps.clamp(0, self.max_seq_len - 1)
        t_emb = self.timestep_embed(timesteps)  # (B, T, H)
        # Broadcast to 3 tokens per step
        t_emb_expanded = t_emb.unsqueeze(2).expand(-1, -1, 3, -1).reshape(batch_size, seq_len * 3, -1)
        tokens = tokens + t_emb_expanded

        tokens = self.embed_dropout(tokens)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len * 3, seq_len * 3, device=device),
            diagonal=1,
        ).bool()

        # Transformer forward
        out = self.transformer(tokens, mask=causal_mask)

        # Extract state token outputs → predict action
        state_outputs = out[:, 1::3]  # (B, T, H)
        action_preds = self.action_head(state_outputs)

        return action_preds

    def get_action(self, states, actions, rtgs, deterministic=False):
        """Get action for the last timestep (inference/RL)."""
        action_preds = self.forward(states, actions, rtgs)
        last_action = action_preds[:, -1]  # (B, action_dim)

        if deterministic:
            return last_action

        std = torch.exp(self.log_std.clamp(-2, 0.5))
        noise = torch.randn_like(last_action) * std
        return torch.clamp(last_action + noise, -1, 1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
