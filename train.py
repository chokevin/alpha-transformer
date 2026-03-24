#!/usr/bin/env python3
"""Decision Transformer for commodity trading.

A transformer model that treats trading as sequence modeling:
  Input:  sequence of (state, action, reward) tuples from past timesteps
  Output: next action (position sizes for each commodity)

Training phases:
  1. Collect trajectories from the commodity environment (random/heuristic policy)
  2. Train transformer on trajectories conditioned on desired returns (SFT)
  3. Fine-tune with RL using sandbox-arena as environment (REINFORCE/GRPO)

This is the architecture behind Decision Transformer (Google Brain, 2021)
applied to multi-commodity trading.

Usage:
    # Phase 1: Collect trajectories
    python train_transformer.py --phase collect --episodes 500

    # Phase 2: Supervised training on collected trajectories
    python train_transformer.py --phase train --epochs 50

    # Phase 3: RL fine-tuning against live environment
    python train_transformer.py --phase rl --episodes 2000

    # Full pipeline
    python train_transformer.py --phase all
"""

import argparse
import json
import math
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Model: Causal Transformer for Trading
# ============================================================

class TradingTransformer(nn.Module):
    """Small causal transformer that predicts trading actions from
    sequences of (state, action, return-to-go).

    Input at each timestep: [state_embed, action_embed, rtg_embed]
    Output: predicted action for current timestep
    """

    def __init__(self, state_dim=36, action_dim=4, hidden_dim=64,
                 n_heads=4, n_layers=3, max_seq_len=60, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.action_dim = action_dim

        # Embeddings for each token type
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.rtg_embed = nn.Linear(1, hidden_dim)  # return-to-go

        # Positional encoding
        self.pos_embed = nn.Embedding(max_seq_len * 3, hidden_dim)  # 3 tokens per step

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # actions in [-1, 1]
        )

        # Log-std for exploration (RL phase)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states, actions, rtgs, timesteps=None):
        """
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
            rtgs: (batch, seq_len, 1)
        Returns:
            action_preds: (batch, seq_len, action_dim)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embed each modality
        s_emb = self.state_embed(states)    # (B, T, H)
        a_emb = self.action_embed(actions)  # (B, T, H)
        r_emb = self.rtg_embed(rtgs)        # (B, T, H)

        # Interleave: [r1, s1, a1, r2, s2, a2, ...]
        tokens = torch.zeros(batch_size, seq_len * 3, self.hidden_dim,
                            device=states.device)
        tokens[:, 0::3] = r_emb
        tokens[:, 1::3] = s_emb
        tokens[:, 2::3] = a_emb

        # Add positional embeddings
        positions = torch.arange(seq_len * 3, device=states.device).unsqueeze(0)
        tokens = tokens + self.pos_embed(positions)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len * 3, seq_len * 3, device=states.device),
            diagonal=1,
        ).bool()

        # Transformer forward
        out = self.transformer(tokens, mask=causal_mask)

        # Extract state token outputs (predict action from state)
        state_outputs = out[:, 1::3]  # (B, T, H)
        action_preds = self.action_head(state_outputs)

        return action_preds

    def get_action(self, states, actions, rtgs, deterministic=False):
        """Get action for the last timestep (for inference/RL)."""
        action_preds = self.forward(states, actions, rtgs)
        last_action = action_preds[:, -1]  # (B, action_dim)

        if deterministic:
            return last_action

        # Add exploration noise
        std = torch.exp(self.log_std.clamp(-2, 0.5))
        noise = torch.randn_like(last_action) * std
        return torch.clamp(last_action + noise, -1, 1)


# ============================================================
# Dataset: Trading trajectories
# ============================================================

class TradingTrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_seq_len=60):
        self.max_seq_len = max_seq_len
        self.samples = []

        for traj in trajectories:
            states = np.array(traj["states"])
            actions = np.array(traj["actions"])
            rewards = np.array(traj["rewards"])

            # Compute return-to-go
            rtgs = np.zeros_like(rewards)
            rtg = 0
            for i in range(len(rewards) - 1, -1, -1):
                rtg = rewards[i] + 0.99 * rtg
                rtgs[i] = rtg

            # Sliding windows
            for start in range(0, len(states) - max_seq_len, max_seq_len // 2):
                end = start + max_seq_len
                self.samples.append({
                    "states": states[start:end].astype(np.float32),
                    "actions": actions[start:end].astype(np.float32),
                    "rtgs": rtgs[start:end].astype(np.float32),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["states"]),
            torch.tensor(s["actions"]),
            torch.tensor(s["rtgs"]).unsqueeze(-1),
        )


# ============================================================
# Phase 1: Collect trajectories
# ============================================================

def collect_trajectories(n_episodes, env_class):
    """Collect trajectories using a random + heuristic mix policy."""
    trajectories = []

    for ep in range(n_episodes):
        env = env_class(n_days=252)
        obs, _ = env.reset()
        states, actions, rewards = [], [], []

        done = False
        while not done:
            states.append(obs.tolist())

            # Mix of random and momentum-based actions
            if np.random.random() < 0.3:
                action = np.random.uniform(-0.5, 0.5, size=4)
            else:
                # Simple momentum: use price vs MA features
                momentum = obs[3:32:8]  # price vs MA5 for each commodity
                action = np.clip(momentum * 5, -1, 1)

            actions.append(action.tolist())
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            done = terminated or truncated

        trajectories.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "total_return": info.get("total_return", 0),
        })

        if (ep + 1) % 50 == 0:
            returns = [t["total_return"] for t in trajectories[-50:]]
            print(f"  Collected {ep+1}/{n_episodes} | "
                  f"avg return: {np.mean(returns):.2f}%")

    return trajectories


# ============================================================
# Phase 2: Supervised training
# ============================================================

def train_supervised(model, trajectories, epochs=50, lr=1e-4, batch_size=32):
    """Train transformer to predict actions from (state, return-to-go) sequences."""
    dataset = TradingTrajectoryDataset(trajectories, max_seq_len=model.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    for epoch in range(1, epochs + 1):
        total_loss = 0
        n_batches = 0

        for states, actions, rtgs in loader:
            pred_actions = model(states, actions, rtgs)
            loss = F.mse_loss(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{epochs} | loss={avg_loss:.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    return model


# ============================================================
# Phase 3: RL fine-tuning
# ============================================================

def rl_finetune(model, env_class, episodes=1000, batch_size=10,
                lr=5e-5, eval_interval=100):
    """Fine-tune transformer with REINFORCE using environment rewards."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for ep in range(0, episodes, batch_size):
        batch_rewards = []
        batch_log_probs = []

        for _ in range(batch_size):
            env = env_class(n_days=252)
            obs, _ = env.reset()

            # Rolling context window
            states_buf = []
            actions_buf = []
            rtgs_buf = []
            rewards_ep = []
            log_probs_ep = []

            target_return = 20.0  # condition on desired 20% return

            done = False
            while not done:
                states_buf.append(obs)

                # Build sequence tensors
                seq_len = min(len(states_buf), model.max_seq_len)
                s = torch.tensor(np.array(states_buf[-seq_len:]), dtype=torch.float32).unsqueeze(0)

                if actions_buf:
                    a = torch.tensor(np.array(actions_buf[-seq_len:]), dtype=torch.float32).unsqueeze(0)
                    # Pad actions to match states length
                    if a.shape[1] < s.shape[1]:
                        pad = torch.zeros(1, s.shape[1] - a.shape[1], model.action_dim)
                        a = torch.cat([pad, a], dim=1)
                else:
                    a = torch.zeros(1, seq_len, model.action_dim)

                # Return-to-go (decreasing as we accumulate rewards)
                current_rtg = target_return - sum(rewards_ep)
                r = torch.full((1, seq_len, 1), current_rtg / 100, dtype=torch.float32)

                # Get action
                with torch.no_grad():
                    mu = model.forward(s, a, r)[:, -1]

                std = torch.exp(model.log_std.clamp(-2, 0.5))
                dist = torch.distributions.Normal(mu, std)
                action_t = dist.sample()
                action_t = torch.clamp(action_t, -1, 1)
                log_prob = dist.log_prob(action_t).sum()

                action = action_t.squeeze(0).detach().numpy()
                actions_buf.append(action)

                obs, reward, terminated, truncated, info = env.step(action)
                rewards_ep.append(reward)
                log_probs_ep.append(log_prob)
                done = terminated or truncated

            # Compute returns
            G = 0
            returns = []
            for r in reversed(rewards_ep):
                G = r + 0.99 * G
                returns.insert(0, G)

            batch_rewards.extend(returns)
            batch_log_probs.extend(log_probs_ep)

        # Policy gradient update
        returns_t = torch.tensor(batch_rewards, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        log_probs_t = torch.stack(batch_log_probs)

        loss = -(log_probs_t * returns_t).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate
        if (ep + batch_size) % eval_interval == 0 or ep == 0:
            eval_returns = []
            for _ in range(20):
                env = env_class(n_days=252)
                obs, _ = env.reset()
                states_buf, actions_buf, rewards_ep = [], [], []
                done = False
                while not done:
                    states_buf.append(obs)
                    seq_len = min(len(states_buf), model.max_seq_len)
                    s = torch.tensor(np.array(states_buf[-seq_len:]),
                                    dtype=torch.float32).unsqueeze(0)
                    a_len = min(len(actions_buf), seq_len)
                    if actions_buf:
                        a = torch.tensor(np.array(actions_buf[-a_len:]),
                                        dtype=torch.float32).unsqueeze(0)
                        if a.shape[1] < s.shape[1]:
                            pad = torch.zeros(1, s.shape[1] - a.shape[1], model.action_dim)
                            a = torch.cat([pad, a], dim=1)
                    else:
                        a = torch.zeros(1, seq_len, model.action_dim)
                    r = torch.full((1, seq_len, 1), 0.2, dtype=torch.float32)

                    with torch.no_grad():
                        action = model.get_action(s, a, r, deterministic=True)
                    action = action.squeeze(0).numpy()
                    actions_buf.append(action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    rewards_ep.append(reward)
                    done = terminated or truncated
                eval_returns.append(info.get("total_return", 0))

            avg_ret = np.mean(eval_returns)
            bar = "█" * int(np.clip((avg_ret + 20) * 1.25, 0, 50))
            bar += "░" * (50 - len(bar))
            print(f"  Ep {ep+batch_size:>5d} | ret={avg_ret:>7.2f}% | "
                  f"loss={loss.item():.4f} | {bar}")

            history.append({
                "episode": ep + batch_size,
                "avg_return": round(avg_ret, 2),
                "loss": round(loss.item(), 4),
            })

    return model, history


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Decision Transformer for trading")
    parser.add_argument("--phase", default="all",
                        choices=["collect", "train", "rl", "all"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--rl-episodes", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=60)
    args = parser.parse_args()

    from envs.commodities import CommodityTradingEnv

    model = TradingTransformer(
        state_dim=36, action_dim=4,
        hidden_dim=args.hidden, n_heads=args.heads,
        n_layers=args.layers, max_seq_len=args.seq_len,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    traj_path = Path("trajectories/transformer_trajectories.json")
    model_path = Path("agents/trading_transformer.pt")

    # Phase 1: Collect
    if args.phase in ("collect", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 1: Collecting {args.episodes} trajectories")
        print(f"{'='*60}")
        trajectories = collect_trajectories(args.episodes, CommodityTradingEnv)
        traj_path.parent.mkdir(exist_ok=True)
        with open(traj_path, "w") as f:
            json.dump(trajectories, f)
        print(f"  Saved to {traj_path}")
    else:
        with open(traj_path) as f:
            trajectories = json.load(f)

    # Phase 2: Supervised training
    if args.phase in ("train", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 2: Supervised training ({args.epochs} epochs)")
        print(f"{'='*60}")
        model = train_supervised(model, trajectories, epochs=args.epochs)
        model_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"  Saved to {model_path}")
    elif model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))

    # Phase 3: RL fine-tuning
    if args.phase in ("rl", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 3: RL fine-tuning ({args.rl_episodes} episodes)")
        print(f"{'='*60}")
        model, history = rl_finetune(model, CommodityTradingEnv,
                                     episodes=args.rl_episodes)
        torch.save(model.state_dict(), model_path)
        print(f"  Saved to {model_path}")

        # Save history
        with open(f"trajectories/transformer_rl_{int(time.time())}.json", "w") as f:
            json.dump(history, f, indent=2)

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Final Evaluation")
    print(f"{'='*60}")

    eval_returns = []
    for _ in range(50):
        env = CommodityTradingEnv(n_days=252)
        obs, _ = env.reset()
        states_buf, actions_buf = [], []
        done = False
        while not done:
            states_buf.append(obs)
            seq_len = min(len(states_buf), model.max_seq_len)
            s = torch.tensor(np.array(states_buf[-seq_len:]),
                            dtype=torch.float32).unsqueeze(0)
            if actions_buf:
                a_len = min(len(actions_buf), seq_len)
                a = torch.tensor(np.array(actions_buf[-a_len:]),
                                dtype=torch.float32).unsqueeze(0)
                if a.shape[1] < s.shape[1]:
                    pad = torch.zeros(1, s.shape[1] - a.shape[1], 4)
                    a = torch.cat([pad, a], dim=1)
            else:
                a = torch.zeros(1, seq_len, 4)
            r = torch.full((1, seq_len, 1), 0.2, dtype=torch.float32)
            with torch.no_grad():
                action = model.get_action(s, a, r, deterministic=True)
            action = action.squeeze(0).numpy()
            actions_buf.append(action)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        eval_returns.append(info.get("total_return", 0))

    # Buy-and-hold baseline
    bh_returns = []
    for _ in range(50):
        env = CommodityTradingEnv(n_days=252)
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(np.ones(4))
            done = terminated or truncated
        bh_returns.append(info.get("total_return", 0))

    print(f"  Transformer:   {np.mean(eval_returns):>7.2f}% ± {np.std(eval_returns):.2f}%")
    print(f"  Buy & Hold:    {np.mean(bh_returns):>7.2f}% ± {np.std(bh_returns):.2f}%")
    print(f"  Alpha:         {np.mean(eval_returns) - np.mean(bh_returns):>+7.2f}%")
    print(f"  Win Rate:      {sum(1 for r in eval_returns if r > 0)/len(eval_returns)*100:.0f}%")
    print(f"  Sharpe (agent):{np.mean(eval_returns)/(np.std(eval_returns)+1e-10)*np.sqrt(1):.3f}")


if __name__ == "__main__":
    main()
