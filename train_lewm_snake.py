#!/usr/bin/env python3
"""LeWM (Le World Model) training for Snake.

Learns a latent world model from trajectories using only two loss terms:
  1. MSE prediction loss: ||ẑ_{t+1} - z_{t+1}||²
  2. SIGReg regularizer: prevents representation collapse

At inference, uses CEM planner to search over action sequences in latent space.

Based on: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
Architecture from Pixels" (Maes et al., 2026)

Usage:
    python train_lewm_snake.py --phase all       # collect + train + eval
    python train_lewm_snake.py --phase train      # train only (reuse data)
    python train_lewm_snake.py --phase eval       # eval only (reuse model)
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
from model.encoder import Encoder
from model.predictor import Predictor
from model.sigreg import SIGReg
from planner import CEMPlanner


# ============================================================
# Dataset: trajectories → (obs, action, next_obs) subsequences
# ============================================================

class LeWMDataset(Dataset):
    """Convert trajectories into fixed-length subsequences for world model training."""

    def __init__(self, trajectories, seq_len=20):
        self.seq_len = seq_len
        self.samples = []

        for traj in trajectories:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.int64)
            T = len(states)

            # Sliding windows
            stride = max(seq_len // 2, 1)
            for start in range(0, max(1, T - seq_len + 1), stride):
                end = min(start + seq_len, T)
                sl = end - start
                if sl < 3:
                    continue
                # Pad to fixed length
                s_pad = np.zeros((seq_len, states.shape[1]), dtype=np.float32)
                a_pad = np.zeros(seq_len, dtype=np.int64)
                s_pad[:sl] = states[start:end]
                a_pad[:sl] = actions[start:end]
                self.samples.append({
                    "states": s_pad,
                    "actions": a_pad,
                    "length": sl,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["states"]),
            torch.tensor(s["actions"]),
            s["length"],
        )


# ============================================================
# Heuristics for data collection (same as train_snake.py)
# ============================================================

def _flood_fill_count(start, blocked, grid_size):
    visited = set()
    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r, c) in visited or (r, c) in blocked:
            continue
        if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
            continue
        visited.add((r, c))
        stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
    return len(visited)


def _bfs_path(start, goal, blocked, grid_size):
    from collections import deque
    queue = deque([(start, [])])
    visited = {start}
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for a, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            if (0 <= nr < grid_size and 0 <= nc < grid_size and
                    (nr, nc) not in visited and (nr, nc) not in blocked):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [a]))
    return None


def _smart_snake(obs, grid_size=10):
    """BFS to food with flood-fill safety."""
    grid = obs.reshape(grid_size, grid_size, 3)
    head_pos = np.argwhere(grid[:, :, 1] == 1.0)
    food_pos = np.argwhere(grid[:, :, 2] == 1.0)
    if len(head_pos) == 0 or len(food_pos) == 0:
        return np.random.randint(4)

    hr, hc = int(head_pos[0][0]), int(head_pos[0][1])
    fr, fc = int(food_pos[0][0]), int(food_pos[0][1])
    body = set(map(tuple, np.argwhere(grid[:, :, 0] == 1.0)))
    snake_len = len(body)
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    safe_moves = []
    for a, (dr, dc) in enumerate(moves):
        nr, nc = hr + dr, hc + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in body:
            safe_moves.append(a)

    if not safe_moves:
        return np.random.randint(4)
    if len(safe_moves) == 1:
        return safe_moves[0]

    path = _bfs_path((hr, hc), (fr, fc), body, grid_size)
    if path:
        first_action = path[0]
        dr, dc = moves[first_action]
        nr, nc = hr + dr, hc + dc
        new_body = body | {(nr, nc)}
        reachable = _flood_fill_count((nr, nc), new_body - {(nr, nc)}, grid_size)
        if reachable >= snake_len:
            return first_action

    best_a, best_reach = safe_moves[0], -1
    for a in safe_moves:
        dr, dc = moves[a]
        nr, nc = hr + dr, hc + dc
        new_body = body | {(nr, nc)}
        reachable = _flood_fill_count((nr, nc), new_body - {(nr, nc)}, grid_size)
        if reachable > best_reach:
            best_reach = reachable
            best_a = a
    return best_a


# ============================================================
# Phase 1: Collect
# ============================================================

def collect_trajectories(n_episodes, grid_size=10, min_score_percentile=50):
    trajectories = []
    for ep in range(n_episodes):
        env = SnakeEnv(grid_size=grid_size)
        obs, _ = env.reset()
        states, actions, rewards = [], [], []
        noise = np.random.choice([0.02, 0.05, 0.1, 0.15], p=[0.3, 0.3, 0.25, 0.15])
        done = False
        while not done:
            states.append(obs.tolist())
            if np.random.random() < noise:
                grid = obs.reshape(grid_size, grid_size, 3)
                head_pos = np.argwhere(grid[:, :, 1] == 1.0)
                body = set(map(tuple, np.argwhere(grid[:, :, 0] == 1.0)))
                mvs = [(-1,0),(0,1),(1,0),(0,-1)]
                if len(head_pos) > 0:
                    hr, hc = head_pos[0]
                    safe = [a for a,(dr,dc) in enumerate(mvs)
                            if 0<=hr+dr<grid_size and 0<=hc+dc<grid_size and (hr+dr,hc+dc) not in body]
                    action = np.random.choice(safe) if safe else np.random.randint(4)
                else:
                    action = np.random.randint(4)
            else:
                action = _smart_snake(obs, grid_size)
            actions.append(int(action))
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
        trajectories.append({
            "states": states, "actions": actions, "rewards": rewards,
            "score": info.get("score", 0),
        })
        if (ep + 1) % 500 == 0:
            recent = trajectories[-500:]
            scores = [t["score"] for t in recent]
            print(f"  Collected {ep+1}/{n_episodes} | avg score: {np.mean(scores):.1f} | max: {max(scores)}")

    scores = [t["score"] for t in trajectories]
    threshold = np.percentile(scores, min_score_percentile)
    filtered = [t for t in trajectories if t["score"] >= threshold]
    print(f"  Filtered: {len(filtered)}/{len(trajectories)} (threshold score={threshold})")
    return filtered


# ============================================================
# Phase 2: Train LeWM
# ============================================================

def train_lewm(encoder, predictor, sigreg, trajectories,
               epochs=100, lr=3e-4, batch_size=32, seq_len=20, lambd=0.1):
    """Train the world model with MSE prediction + SIGReg."""
    dataset = LeWMDataset(trajectories, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    warmup = min(5, epochs // 5)
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")
    print(f"  Loss: MSE + {lambd} × SIGReg")

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        encoder.train()
        predictor.train()
        total_pred_loss = 0
        total_sig_loss = 0
        n_batches = 0

        for states, actions, lengths in loader:
            # Encode all observations
            z = encoder(states)  # (B, T, D)

            # Predict next states
            z_pred = predictor(z, actions)  # (B, T, D)

            # Prediction loss: MSE(ẑ_{t+1}, z_{t+1}) for t=1..T-1
            pred_loss = F.mse_loss(z_pred[:, :-1], z[:, 1:])

            # SIGReg on encoder embeddings (step-wise as in paper)
            z_flat = z.reshape(-1, z.shape[-1])
            sig_loss = sigreg(z_flat)

            loss = pred_loss + lambd * sig_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_pred_loss += pred_loss.item()
            total_sig_loss += sig_loss.item()
            n_batches += 1

        scheduler.step()
        avg_pred = total_pred_loss / max(n_batches, 1)
        avg_sig = total_sig_loss / max(n_batches, 1)
        avg_total = avg_pred + lambd * avg_sig

        if avg_total < best_loss:
            best_loss = avg_total

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{epochs} | pred={avg_pred:.6f} | "
                  f"sigreg={avg_sig:.6f} | total={avg_total:.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")


# ============================================================
# Evaluation with planner
# ============================================================

def evaluate_with_planner(encoder, predictor, n_episodes=50, grid_size=10,
                          horizon=8, n_samples=256, n_iterations=3):
    """Evaluate world model + CEM planner on Snake."""
    planner = CEMPlanner(
        encoder, predictor,
        horizon=horizon, n_samples=n_samples, n_iterations=n_iterations,
        top_k=32, n_actions=4, discrete=True,
    )

    scores = []
    for ep in range(n_episodes):
        env = SnakeEnv(grid_size=grid_size)
        obs, _ = env.reset()

        # Goal: the food location encoded as observation
        # We'll re-encode the goal each step since food moves
        done = False
        while not done:
            # Create a "goal" observation: current obs but with food highlighted
            # The planner minimizes distance to the food's latent encoding
            goal_obs = obs.copy()  # food is already in the observation

            action = planner.plan(obs, goal_obs=goal_obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        scores.append(info.get("score", 0))
        planner.reset()

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} | "
                  f"avg: {np.mean(scores):.1f} | last: {scores[-1]}")

    return scores


def evaluate_heuristic(n_episodes=50, grid_size=10):
    """Run baselines for comparison."""
    results = {}

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

    smart_scores = []
    for _ in range(n_episodes):
        env = SnakeEnv(grid_size=grid_size)
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, t, tr, info = env.step(_smart_snake(obs, grid_size))
            done = t or tr
        smart_scores.append(info.get("score", 0))
    results["Smart BFS"] = smart_scores

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LeWM for Snake")
    parser.add_argument("--phase", default="all",
                        choices=["collect", "train", "eval", "all"])
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--pred-layers", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambd", type=float, default=0.1,
                        help="SIGReg weight (only hyperparameter)")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--min-score-percentile", type=int, default=30)
    # Planner
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()

    obs_dim = args.grid_size ** 2 * 3  # 300 for 10x10

    encoder = Encoder(obs_dim=obs_dim, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    predictor = Predictor(embed_dim=args.embed_dim, action_dim=4,
                          hidden_dim=args.embed_dim, n_layers=args.pred_layers,
                          n_heads=args.pred_heads, discrete_actions=True)
    sigreg = SIGReg(embed_dim=args.embed_dim)

    enc_params = sum(p.numel() for p in encoder.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    print(f"LeWM: encoder={enc_params:,} + predictor={pred_params:,} = {enc_params+pred_params:,} params")

    traj_path = Path("results/lewm_snake_trajectories.json")
    model_path = Path("results/lewm_snake.pt")

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
            print(f"  No trajectories at {traj_path}. Run --phase collect first.")
            return

    # Phase 2: Train
    if args.phase in ("train", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 2: Training LeWM ({args.epochs} epochs, λ={args.lambd})")
        print(f"{'='*60}")
        train_lewm(encoder, predictor, sigreg, trajectories,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    seq_len=args.seq_len, lambd=args.lambd)
        model_path.parent.mkdir(exist_ok=True)
        torch.save({
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "config": {"obs_dim": obs_dim, "embed_dim": args.embed_dim,
                       "hidden_dim": args.hidden_dim, "pred_layers": args.pred_layers,
                       "pred_heads": args.pred_heads},
        }, model_path)
        print(f"  Saved to {model_path}")
    elif model_path.exists():
        ckpt = torch.load(model_path, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        predictor.load_state_dict(ckpt["predictor"])
        print(f"  Loaded model from {model_path}")

    # Phase 3: Evaluate
    if args.phase in ("eval", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 3: Evaluation ({args.eval_episodes} episodes)")
        print(f"{'='*60}")

        encoder.eval()
        predictor.eval()

        print("\n  LeWM + CEM Planner:")
        planner_scores = evaluate_with_planner(
            encoder, predictor, n_episodes=args.eval_episodes,
            grid_size=args.grid_size, horizon=args.horizon,
            n_samples=args.n_samples,
        )

        print("\n  Baselines:")
        baselines = evaluate_heuristic(args.eval_episodes, args.grid_size)

        print(f"\n  {'Model':<20} {'Avg Score':>10} {'Max':>5}")
        print(f"  {'─'*40}")
        print(f"  {'LeWM + CEM':<20} {np.mean(planner_scores):>10.2f} {max(planner_scores):>5d}")
        for name, scores in baselines.items():
            print(f"  {name:<20} {np.mean(scores):>10.2f} {max(scores):>5d}")
        print(f"  {'─'*40}")
        print(f"  {'vs Random':<20} {np.mean(planner_scores) - np.mean(baselines['Random']):>+10.2f}")


if __name__ == "__main__":
    main()
