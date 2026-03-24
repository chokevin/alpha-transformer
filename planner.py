"""MPC Planner: searches over action sequences using the world model.

At each timestep:
1. Encode current observation → z_now
2. Sample N random action sequences (horizon H)
3. Rollout each through the world model
4. Score each by a cost function (e.g., distance to goal, predicted return)
5. Execute the first action of the best sequence
6. Repeat next timestep
"""

import torch
import numpy as np
from typing import Callable, Optional


class MPCPlanner:
    """Model Predictive Control planner using the world model."""

    def __init__(self, world_model, horizon: int = 10, n_samples: int = 512,
                 n_iterations: int = 3, top_k: int = 64,
                 action_dim: int = 4, action_range: float = 1.0):
        """
        Args:
            world_model: trained MarketWorldModel
            horizon: planning horizon (steps to look ahead)
            n_samples: number of action sequences to sample
            n_iterations: CEM iterations (refine around best candidates)
            top_k: number of top candidates to keep each iteration
            action_dim: dimension of action space
            action_range: max absolute action value
        """
        self.model = world_model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.top_k = top_k
        self.action_dim = action_dim
        self.action_range = action_range

        # Running mean/std for CEM (Cross-Entropy Method)
        self._mean = None
        self._std = None

    def plan(self, obs: np.ndarray, cost_fn: Callable) -> np.ndarray:
        """Plan the best action for the current observation.

        Args:
            obs: current observation (obs_dim,)
            cost_fn: function(z_trajectory) → cost (lower is better)
                     z_trajectory: (N, H+1, embed_dim)
        Returns:
            best_action: (action_dim,)
        """
        device = next(self.model.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # Encode current state
        with torch.no_grad():
            z_now = self.model.encode(obs_t).unsqueeze(1)  # (1, 1, D)

        # Initialize CEM distribution
        mean = self._mean if self._mean is not None else torch.zeros(
            self.horizon, self.action_dim, device=device)
        std = self._std if self._std is not None else torch.ones(
            self.horizon, self.action_dim, device=device) * 0.5

        best_actions = None
        best_cost = float('inf')

        for iteration in range(self.n_iterations):
            # Sample action sequences from current distribution
            noise = torch.randn(self.n_samples, self.horizon, self.action_dim,
                               device=device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            actions = actions.clamp(-self.action_range, self.action_range)

            # Expand z_now for all samples
            z_init = z_now.expand(self.n_samples, -1, -1)

            # Rollout all candidates through world model
            with torch.no_grad():
                z_trajectories = self.model.rollout(z_init, actions)

            # Score each trajectory
            costs = cost_fn(z_trajectories)  # (N,)

            # Select top-k
            top_indices = costs.argsort()[:self.top_k]
            top_actions = actions[top_indices]

            # Update CEM distribution
            mean = top_actions.mean(dim=0)
            std = top_actions.std(dim=0).clamp(min=0.05)

            # Track best
            if costs[top_indices[0]] < best_cost:
                best_cost = costs[top_indices[0]]
                best_actions = actions[top_indices[0]]

        # Shift mean for next timestep (warm start)
        self._mean = torch.cat([mean[1:], torch.zeros(1, self.action_dim, device=device)])
        self._std = torch.cat([std[1:], torch.ones(1, self.action_dim, device=device) * 0.5])

        # Return first action of best sequence
        return best_actions[0].cpu().numpy()

    def reset(self):
        """Reset planner state (call at start of new episode)."""
        self._mean = None
        self._std = None


def make_return_cost_fn(world_model, target_return: float = 0.2):
    """Cost function: prefer trajectories whose final latent is far from initial.

    A simple proxy for return: the model should move the portfolio state
    away from the starting point in a direction consistent with positive returns.

    For a more sophisticated cost, you'd train a small reward predictor
    on top of the latent space.
    """
    def cost_fn(z_trajectories):
        # z_trajectories: (N, H+1, D)
        z_start = z_trajectories[:, 0]   # (N, D)
        z_end = z_trajectories[:, -1]     # (N, D)

        # Cost = negative magnitude of latent change (we want big changes)
        # Plus variance penalty (we want consistent trajectories)
        change = (z_end - z_start).pow(2).sum(dim=-1)  # (N,)

        # Also penalize trajectories that are too volatile
        diffs = torch.diff(z_trajectories, dim=1)  # (N, H, D)
        volatility = diffs.pow(2).sum(dim=-1).mean(dim=-1)  # (N,)

        return -change + 0.1 * volatility  # lower is better

    return cost_fn
