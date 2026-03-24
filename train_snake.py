#!/usr/bin/env python3
"""Decision Transformer for Snake game.

Trains a causal transformer to play Snake using the same architecture
as the commodity trader, but with discrete actions and cross-entropy loss.

This is a proven domain for Decision Transformers — the reward signal is
dense and deterministic (eat food = +1, die = -1), making it much easier
to learn from than noisy financial data.

Usage:
    python train_snake.py --phase all             # full pipeline
    python train_snake.py --phase collect --episodes 5000
    python train_snake.py --phase train --epochs 50
    python train_snake.py --phase rl --rl-episodes 2000
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

from envs.snake import SnakeEnv

STATE_DIM = 300  # 10x10x3
N_ACTIONS = 4    # up, right, down, left


# ============================================================
# Model: Decision Transformer for discrete actions
# ============================================================

class SnakeTransformer(nn.Module):
    """Decision Transformer for Snake with discrete action output."""

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden_dim=128,
                 n_heads=4, n_layers=4, max_seq_len=50, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.n_actions = n_actions

        # Token embeddings
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.action_embed = nn.Embedding(n_actions, hidden_dim)
        self.rtg_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim),
        )

        self.pos_embed = nn.Embedding(max_seq_len * 3, hidden_dim)
        self.timestep_embed = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embed_dropout = nn.Dropout(dropout)

        # Discrete action head (logits)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self._init_weights()

    def _init_weights(self):
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
            states:  (B, T, state_dim)
            actions: (B, T) long tensor of action indices
            rtgs:    (B, T, 1)
        Returns:
            logits: (B, T, n_actions)
        """
        B, T = states.shape[0], states.shape[1]
        device = states.device

        s_emb = self.state_embed(states)
        a_emb = self.action_embed(actions)
        r_emb = self.rtg_embed(rtgs)

        tokens = torch.zeros(B, T * 3, self.hidden_dim, device=device)
        tokens[:, 0::3] = r_emb
        tokens[:, 1::3] = s_emb
        tokens[:, 2::3] = a_emb

        positions = torch.arange(T * 3, device=device).unsqueeze(0)
        tokens = tokens + self.pos_embed(positions)

        if timesteps is None:
            timesteps = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        timesteps = timesteps.clamp(0, self.max_seq_len - 1)
        t_emb = self.timestep_embed(timesteps).unsqueeze(2).expand(-1, -1, 3, -1).reshape(B, T * 3, -1)
        tokens = self.embed_dropout(tokens + t_emb)

        causal_mask = torch.triu(torch.ones(T * 3, T * 3, device=device), diagonal=1).bool()
        out = self.transformer(tokens, mask=causal_mask)

        state_outputs = out[:, 1::3]
        logits = self.action_head(state_outputs)
        return logits

    def get_action(self, states, actions, rtgs, deterministic=False):
        """Get action for the last timestep."""
        logits = self.forward(states, actions, rtgs)
        last_logits = logits[:, -1]
        if deterministic:
            return last_logits.argmax(dim=-1)
        probs = F.softmax(last_logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Dataset
# ============================================================

class SnakeTrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_seq_len=50, gamma=0.99):
        self.max_seq_len = max_seq_len
        self.samples = []

        for traj in trajectories:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.int64)
            rewards = np.array(traj["rewards"], dtype=np.float32)

            rtgs = np.zeros_like(rewards)
            rtg = 0
            for i in range(len(rewards) - 1, -1, -1):
                rtg = rewards[i] + gamma * rtg
                rtgs[i] = rtg

            stride = max(max_seq_len // 2, 1)
            for start in range(0, max(1, len(states) - max_seq_len + 1), stride):
                end = min(start + max_seq_len, len(states))
                # Pad to fixed length
                sl = end - start
                s_pad = np.zeros((max_seq_len, states.shape[1]), dtype=np.float32)
                a_pad = np.zeros(max_seq_len, dtype=np.int64)
                r_pad = np.zeros(max_seq_len, dtype=np.float32)
                s_pad[:sl] = states[start:end]
                a_pad[:sl] = actions[start:end]
                r_pad[:sl] = rtgs[start:end]
                self.samples.append({
                    "states": s_pad,
                    "actions": a_pad,
                    "rtgs": r_pad,
                    "length": sl,
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
# Heuristic policies for data collection
# ============================================================

def _greedy_snake(obs, grid_size=10):
    """Move toward food, avoid walls and self."""
    grid = obs.reshape(grid_size, grid_size, 3)
    head_pos = np.argwhere(grid[:, :, 1] == 1.0)
    food_pos = np.argwhere(grid[:, :, 2] == 1.0)

    if len(head_pos) == 0 or len(food_pos) == 0:
        return np.random.randint(4)

    hr, hc = head_pos[0]
    fr, fc = food_pos[0]
    body = set(map(tuple, np.argwhere(grid[:, :, 0] == 1.0)))

    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
    safe = []
    for a, (dr, dc) in enumerate(moves):
        nr, nc = hr + dr, hc + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in body:
            safe.append(a)

    if not safe:
        return np.random.randint(4)

    # Pick safe move closest to food
    best_a = safe[0]
    best_dist = float("inf")
    for a in safe:
        dr, dc = moves[a]
        nr, nc = hr + dr, hc + dc
        dist = abs(nr - fr) + abs(nc - fc)
        if dist < best_dist:
            best_dist = dist
            best_a = a

    return best_a


def _noisy_greedy(obs, grid_size=10, noise=0.2):
    """Greedy with random exploration."""
    if np.random.random() < noise:
        return np.random.randint(4)
    return _greedy_snake(obs, grid_size)


# ============================================================
# Phase 1: Collect
# ============================================================

def collect_trajectories(n_episodes, grid_size=10, min_score_percentile=50):
    """Collect trajectories using greedy + noisy heuristics."""
    trajectories = []

    for ep in range(n_episodes):
        env = SnakeEnv(grid_size=grid_size)
        obs, _ = env.reset()
        states, actions, rewards = [], [], []

        # Vary exploration noise
        noise = np.random.uniform(0.05, 0.3)
        done = False
        while not done:
            states.append(obs.tolist())
            action = _noisy_greedy(obs, grid_size, noise)
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            done = terminated or truncated

        trajectories.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "score": info.get("score", 0),
        })

        if (ep + 1) % 500 == 0:
            recent = trajectories[-500:]
            scores = [t["score"] for t in recent]
            print(f"  Collected {ep+1}/{n_episodes} | "
                  f"avg score: {np.mean(scores):.1f} | "
                  f"max: {np.max(scores)} | "
                  f"avg len: {np.mean([len(t['states']) for t in recent]):.0f}")

    # Filter top trajectories
    scores = [t["score"] for t in trajectories]
    threshold = np.percentile(scores, min_score_percentile)
    filtered = [t for t in trajectories if t["score"] >= threshold]
    print(f"  Filtered: {len(filtered)}/{len(trajectories)} trajectories "
          f"(top {100-min_score_percentile}%, threshold score={threshold})")

    return filtered


# ============================================================
# Phase 2: SFT
# ============================================================

def train_supervised(model, trajectories, epochs=50, lr=3e-4, batch_size=64):
    """Train transformer with cross-entropy loss on action prediction."""
    dataset = SnakeTrajectoryDataset(trajectories, max_seq_len=model.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup_epochs = min(5, epochs // 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_acc = 0
        n_batches = 0

        for states, actions, rtgs in loader:
            logits = model(states, actions, rtgs)  # (B, T, n_actions)
            logits_flat = logits.reshape(-1, model.n_actions)
            targets_flat = actions.reshape(-1)

            loss = F.cross_entropy(logits_flat, targets_flat)
            acc = (logits_flat.argmax(-1) == targets_flat).float().mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"acc={avg_acc:.3f} | lr={scheduler.get_last_lr()[0]:.6f}")

    return model


# ============================================================
# Evaluation
# ============================================================

def run_episode(model, env, target_score=10, deterministic=True):
    """Run single episode, return info + trajectory."""
    obs, _ = env.reset()
    states_buf, actions_buf, rewards_buf = [], [], []
    done = False

    while not done:
        states_buf.append(obs)
        seq_len = min(len(states_buf), model.max_seq_len)

        s = torch.tensor(np.array(states_buf[-seq_len:]),
                         dtype=torch.float32).unsqueeze(0)

        if actions_buf:
            a_len = min(len(actions_buf), seq_len)
            a = torch.tensor(actions_buf[-a_len:], dtype=torch.long).unsqueeze(0)
            if a.shape[1] < s.shape[1]:
                pad = torch.zeros(1, s.shape[1] - a.shape[1], dtype=torch.long)
                a = torch.cat([pad, a], dim=1)
        else:
            a = torch.zeros(1, seq_len, dtype=torch.long)

        current_rtg = target_score - sum(rewards_buf)
        r = torch.full((1, seq_len, 1), current_rtg, dtype=torch.float32)

        with torch.no_grad():
            action = model.get_action(s, a, r, deterministic=deterministic)
        action = action.item()

        actions_buf.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_buf.append(reward)
        done = terminated or truncated

    return info, {"states": states_buf, "actions": actions_buf, "rewards": rewards_buf}


def evaluate(model, n_episodes=100, grid_size=10, target_score=10):
    """Evaluate model over many episodes."""
    scores = []
    lengths = []
    model.eval()
    for _ in range(n_episodes):
        env = SnakeEnv(grid_size=grid_size)
        info, traj = run_episode(model, env, target_score=target_score)
        scores.append(info.get("score", 0))
        lengths.append(len(traj["states"]))
    return scores, lengths


# ============================================================
# Baselines
# ============================================================

def run_baselines(n_episodes=100, grid_size=10):
    """Run baseline strategies."""
    results = {}

    # Random
    rand_scores = []
    for _ in range(n_episodes):
        env = SnakeEnv(grid_size=grid_size)
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, t, tr, info = env.step(env.action_space.sample())
            done = t or tr
        rand_scores.append(info.get("score", 0))
    results["Random"] = rand_scores

    # Greedy heuristic
    greedy_scores = []
    for _ in range(n_episodes):
        env = SnakeEnv(grid_size=grid_size)
        obs, _ = env.reset()
        done = False
        while not done:
            action = _greedy_snake(obs, grid_size)
            obs, _, t, tr, info = env.step(action)
            done = t or tr
        greedy_scores.append(info.get("score", 0))
    results["Greedy"] = greedy_scores

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Decision Transformer for Snake")
    parser.add_argument("--phase", default="all",
                        choices=["collect", "train", "rl", "all"])
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--target-score", type=int, default=10)
    parser.add_argument("--min-score-percentile", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    model = SnakeTransformer(
        state_dim=args.grid_size ** 2 * 3,
        n_actions=N_ACTIONS,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        max_seq_len=args.seq_len,
    )
    print(f"Model: {model.param_count():,} parameters")

    traj_path = Path("results/snake_trajectories.json")
    model_path = Path("results/snake_transformer.pt")

    # Phase 1: Collect
    if args.phase in ("collect", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 1: Collecting {args.episodes} trajectories")
        print(f"{'='*60}")
        trajectories = collect_trajectories(
            args.episodes, grid_size=args.grid_size,
            min_score_percentile=args.min_score_percentile,
        )
        traj_path.parent.mkdir(exist_ok=True)
        with open(traj_path, "w") as f:
            json.dump(trajectories, f)
        print(f"  Saved {len(trajectories)} trajectories")
    else:
        if traj_path.exists():
            with open(traj_path) as f:
                trajectories = json.load(f)
            print(f"  Loaded {len(trajectories)} trajectories")
        else:
            print(f"  No trajectories found. Run --phase collect first.")
            return

    # Phase 2: Supervised training
    if args.phase in ("train", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 2: Supervised training ({args.epochs} epochs)")
        print(f"{'='*60}")
        model = train_supervised(model, trajectories, epochs=args.epochs,
                                 lr=args.lr, batch_size=args.batch_size)
        model_path.parent.mkdir(exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {"state_dim": args.grid_size ** 2 * 3,
                       "n_actions": N_ACTIONS, "hidden_dim": args.hidden,
                       "n_layers": args.layers, "max_seq_len": args.seq_len},
        }, model_path)
        print(f"  Saved to {model_path}")
    elif model_path.exists():
        ckpt = torch.load(model_path, weights_only=False)
        model = SnakeTransformer(**ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded model from {model_path}")

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Final Evaluation (200 episodes)")
    print(f"{'='*60}")

    scores, lengths = evaluate(model, 200, args.grid_size, args.target_score)
    baselines = run_baselines(200, args.grid_size)

    print(f"\n  {'Model':<20} {'Avg Score':>10} {'Max':>5} {'Avg Len':>8}")
    print(f"  {'─'*45}")
    print(f"  {'Transformer':<20} {np.mean(scores):>10.2f} {np.max(scores):>5d} "
          f"{np.mean(lengths):>8.1f}")
    for name, bscores in baselines.items():
        print(f"  {name:<20} {np.mean(bscores):>10.2f} {np.max(bscores):>5d}")
    print(f"  {'─'*45}")
    improvement = np.mean(scores) - np.mean(baselines["Random"])
    print(f"  {'vs Random':<20} {improvement:>+10.2f}")
    improvement_g = np.mean(scores) - np.mean(baselines["Greedy"])
    print(f"  {'vs Greedy':<20} {improvement_g:>+10.2f}")


if __name__ == "__main__":
    main()
