"""MPC Planner using CEM for the LeWM world model.

Uses Cross-Entropy Method to search over action sequences in latent space.
Supports both goal-based planning (Snake: reach food) and return-based
planning (commodities: maximize returns).

For discrete actions, samples categorical distributions.
For continuous actions, samples Gaussian distributions.
"""

import torch
import numpy as np


class CEMPlanner:
    """Cross-Entropy Method planner for latent world models.

    At each step:
    1. Encode current obs → z
    2. Sample N action sequences of length H
    3. Roll out each through predictor in latent space
    4. Score by cost function
    5. Keep top-K, refit distribution, repeat
    6. Execute first action of best sequence
    """

    def __init__(self, encoder, predictor, horizon: int = 10,
                 n_samples: int = 512, n_iterations: int = 3, top_k: int = 64,
                 n_actions: int = 4, discrete: bool = True):
        self.encoder = encoder
        self.predictor = predictor
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.top_k = top_k
        self.n_actions = n_actions
        self.discrete = discrete

    @torch.no_grad()
    def plan(self, obs: np.ndarray, goal_obs: np.ndarray = None,
             cost_fn=None) -> int:
        """Plan the best action for current observation.

        Args:
            obs: current observation
            goal_obs: goal observation (for goal-based planning)
            cost_fn: custom cost function(z_trajectory) → costs (N,)

        Returns:
            best_action: scalar (discrete) or array (continuous)
        """
        device = next(self.encoder.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        self.encoder.eval()
        self.predictor.eval()

        z_now = self.encoder(obs_t)  # (1, D)

        # Encode goal if provided
        z_goal = None
        if goal_obs is not None:
            goal_t = torch.tensor(goal_obs, dtype=torch.float32, device=device).unsqueeze(0)
            z_goal = self.encoder(goal_t)  # (1, D)

        if self.discrete:
            return self._plan_discrete(z_now, z_goal, cost_fn, device)
        else:
            return self._plan_continuous(z_now, z_goal, cost_fn, device)

    def _plan_discrete(self, z_now, z_goal, cost_fn, device):
        """CEM for discrete action spaces."""
        H = self.horizon
        N = self.n_samples

        # Initialize uniform action probabilities
        probs = torch.ones(H, self.n_actions, device=device) / self.n_actions

        best_actions = None
        best_cost = float("inf")

        for iteration in range(self.n_iterations):
            # Sample action sequences from categorical
            action_seqs = torch.multinomial(
                probs.unsqueeze(0).expand(N, -1, -1).reshape(N * H, -1),
                1
            ).reshape(N, H).squeeze(-1)  # (N, H)

            # Rollout in latent space
            z = z_now.expand(N, -1)  # (N, D)
            z_traj = [z]
            for t in range(H):
                z = self.predictor.predict_step(z, action_seqs[:, t])
                z_traj.append(z)
            z_traj = torch.stack(z_traj, dim=1)  # (N, H+1, D)

            # Compute costs
            if cost_fn is not None:
                costs = cost_fn(z_traj)
            elif z_goal is not None:
                # Goal-based: minimize distance to goal at end of rollout
                costs = (z_traj[:, -1] - z_goal).pow(2).sum(dim=-1)
            else:
                raise ValueError("Must provide either goal_obs or cost_fn")

            # Select top-K
            top_idx = costs.argsort()[:self.top_k]
            top_actions = action_seqs[top_idx]  # (K, H)

            # Update probabilities from top-K empirical distribution
            for t in range(H):
                counts = torch.zeros(self.n_actions, device=device)
                for a in range(self.n_actions):
                    counts[a] = (top_actions[:, t] == a).float().sum()
                probs[t] = (counts + 1) / (self.top_k + self.n_actions)  # Laplace smoothing

            # Track best
            if costs[top_idx[0]] < best_cost:
                best_cost = costs[top_idx[0]]
                best_actions = action_seqs[top_idx[0]]

        return best_actions[0].item()

    def _plan_continuous(self, z_now, z_goal, cost_fn, device):
        """CEM for continuous action spaces."""
        H = self.horizon
        N = self.n_samples
        A = self.n_actions

        mean = torch.zeros(H, A, device=device)
        std = torch.ones(H, A, device=device) * 0.5

        best_actions = None
        best_cost = float("inf")

        for iteration in range(self.n_iterations):
            noise = torch.randn(N, H, A, device=device)
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1, 1)

            # Rollout
            z = z_now.expand(N, -1)
            z_traj = [z]
            for t in range(H):
                z = self.predictor.predict_step(z, actions[:, t])
                z_traj.append(z)
            z_traj = torch.stack(z_traj, dim=1)

            if cost_fn is not None:
                costs = cost_fn(z_traj)
            elif z_goal is not None:
                costs = (z_traj[:, -1] - z_goal).pow(2).sum(dim=-1)
            else:
                raise ValueError("Must provide either goal_obs or cost_fn")

            top_idx = costs.argsort()[:self.top_k]
            top_actions = actions[top_idx]
            mean = top_actions.mean(dim=0)
            std = top_actions.std(dim=0).clamp(min=0.05)

            if costs[top_idx[0]] < best_cost:
                best_cost = costs[top_idx[0]]
                best_actions = actions[top_idx[0]]

        return best_actions[0].cpu().numpy()

    def reset(self):
        """Reset planner state between episodes."""
        pass
