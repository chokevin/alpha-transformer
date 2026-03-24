"""World Model: Encoder + Predictor + SIGReg.

Combines all components into a single module for training and planning.
"""

import torch
import torch.nn as nn

from model.encoder import MarketEncoder
from model.predictor import MarketPredictor
from model.sigreg import SIGReg


class MarketWorldModel(nn.Module):
    """LeWM-style world model for commodity trading.

    Training: encode observations → predict next latent → MSE + SIGReg loss
    Planning: rollout action sequences through predictor → pick best
    """

    def __init__(self, obs_dim=36, action_dim=4, embed_dim=192,
                 hidden_dim=256, pred_hidden=192, n_layers=4, n_heads=4,
                 num_proj=512, sigreg_knots=17, max_seq_len=10):
        super().__init__()

        self.encoder = MarketEncoder(obs_dim, embed_dim, hidden_dim)
        self.predictor = MarketPredictor(
            embed_dim, action_dim, pred_hidden,
            n_layers, n_heads, max_seq_len=max_seq_len,
        )
        self.sigreg = SIGReg(embed_dim, num_proj, sigreg_knots)
        self.embed_dim = embed_dim
        self.action_dim = action_dim

    def encode(self, obs):
        """Encode observations into latent space."""
        return self.encoder(obs)

    def predict(self, z_seq, action_seq):
        """Predict next latent states."""
        return self.predictor(z_seq, action_seq)

    def training_step(self, obs_seq, action_seq, sigreg_weight=0.09):
        """One training step.

        Args:
            obs_seq: (B, T, obs_dim) — observation sequence
            action_seq: (B, T, action_dim) — action sequence
            sigreg_weight: λ for SIGReg loss

        Returns:
            loss, pred_loss, sigreg_loss
        """
        # Encode all observations
        z_seq = self.encode(obs_seq)  # (B, T, embed_dim)

        # Predict: given z[0:T-1] and actions[0:T-1], predict z[1:T]
        z_input = z_seq[:, :-1]
        z_target = z_seq[:, 1:]
        actions = action_seq  # action_seq already has T-1 entries (one per obs transition)

        z_pred = self.predict(z_input, actions)

        # Losses
        pred_loss = (z_pred - z_target.detach()).pow(2).mean()
        sig_loss = self.sigreg(z_seq)
        loss = pred_loss + sigreg_weight * sig_loss

        return loss, pred_loss, sig_loss

    @torch.no_grad()
    def rollout(self, z_init, action_sequence):
        """Rollout predictor given initial latent and action sequence.

        Args:
            z_init: (B, 1, embed_dim) — initial latent state
            action_sequence: (B, T, action_dim) — planned actions

        Returns:
            z_trajectory: (B, T+1, embed_dim) — predicted latent trajectory
        """
        B, T, _ = action_sequence.shape
        z = z_init
        trajectory = [z]

        for t in range(T):
            action_t = action_sequence[:, t:t+1]
            z_pred = self.predict(z[:, -1:], action_t)
            z = torch.cat([z, z_pred], dim=1)
            trajectory.append(z_pred)

        return torch.cat(trajectory, dim=1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
