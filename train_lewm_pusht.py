#!/usr/bin/env python3
"""LeWM training on Push-T: the full collect → train → eval loop.

This demonstrates the value of distributed data collection:
  1. Collect Push-T trajectories (locally or via AKS)
  2. Train LeWM world model (encoder + predictor + SIGReg)
  3. Evaluate with CEM planner against baselines

Usage:
    # Full pipeline (collect locally + train + eval)
    python train_lewm_pusht.py --phase all --episodes 500

    # Train on pre-collected HDF5 data
    python train_lewm_pusht.py --phase train --data results/pusht_500.h5

    # Eval only
    python train_lewm_pusht.py --phase eval
"""

import argparse
import json
import math
import time
import h5py
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.encoder import Encoder
from model.predictor import Predictor
from model.sigreg import SIGReg


# Push-T constants
STATE_DIM = 5   # [agent_x, agent_y, block_x, block_y, block_angle]
ACTION_DIM = 2  # [target_x, target_y] continuous [0, 512]
EMBED_DIM = 64
HIDDEN_DIM = 128


# ============================================================
# Dataset
# ============================================================

class PushTDataset(Dataset):
    """Load Push-T trajectories from HDF5 into fixed-length subsequences."""

    def __init__(self, h5_path, seq_len=20):
        self.seq_len = seq_len
        self.samples = []

        with h5py.File(h5_path, "r") as f:
            states = f["agent_pos"][:]       # (N, 2) — but we want full state
            actions = f["action"][:]          # (N, 2)
            episode_idx = f["episode_idx"][:]
            step_idx = f["step_idx"][:]

            # If we have full 5-dim state from JSON collection
            if "pixels" not in f:
                obs = states
            else:
                obs = states  # Use agent_pos as state for now

        # Group by episode
        episodes = {}
        for i in range(len(obs)):
            ep = int(episode_idx[i])
            if ep not in episodes:
                episodes[ep] = {"obs": [], "actions": []}
            episodes[ep]["obs"].append(obs[i])
            episodes[ep]["actions"].append(actions[i])

        # Create sliding windows
        for ep_data in episodes.values():
            ep_obs = np.array(ep_data["obs"], dtype=np.float32)
            ep_act = np.array(ep_data["actions"], dtype=np.float32)
            T = len(ep_obs)

            stride = max(seq_len // 2, 1)
            for start in range(0, max(1, T - seq_len + 1), stride):
                end = min(start + seq_len, T)
                sl = end - start
                if sl < 3:
                    continue
                s_pad = np.zeros((seq_len, ep_obs.shape[1]), dtype=np.float32)
                a_pad = np.zeros((seq_len, ep_act.shape[1]), dtype=np.float32)
                s_pad[:sl] = ep_obs[start:end]
                a_pad[:sl] = ep_act[start:end]
                self.samples.append({"obs": s_pad, "actions": a_pad, "length": sl})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.tensor(s["obs"]), torch.tensor(s["actions"]), s["length"]


class PushTJSONDataset(Dataset):
    """Load Push-T trajectories from JSON."""

    def __init__(self, json_path, seq_len=20):
        self.seq_len = seq_len
        self.samples = []

        with open(json_path) as f:
            trajectories = json.load(f)

        for traj in trajectories:
            states = np.array(traj["states"], dtype=np.float32)  # (T, 2) agent_pos
            actions = np.array(traj["actions"], dtype=np.float32)  # (T, 2)
            T = len(states)

            stride = max(seq_len // 2, 1)
            for start in range(0, max(1, T - seq_len + 1), stride):
                end = min(start + seq_len, T)
                sl = end - start
                if sl < 3:
                    continue
                s_pad = np.zeros((seq_len, states.shape[1]), dtype=np.float32)
                a_pad = np.zeros((seq_len, actions.shape[1]), dtype=np.float32)
                s_pad[:sl] = states[start:end]
                a_pad[:sl] = actions[start:end]
                self.samples.append({"obs": s_pad, "actions": a_pad, "length": sl})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.tensor(s["obs"]), torch.tensor(s["actions"]), s["length"]


# ============================================================
# Training
# ============================================================

def train_lewm(encoder, predictor, sigreg, dataset,
               epochs=100, lr=3e-4, batch_size=64, lambd=0.1):
    """Train LeWM: MSE prediction + SIGReg."""
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
        total_pred, total_sig = 0, 0
        n = 0

        for obs, actions, lengths in loader:
            # Normalize actions to [-1, 1] from [0, 512]
            actions_norm = actions / 256.0 - 1.0

            z = encoder(obs)
            z_pred = predictor(z, actions_norm)

            pred_loss = F.mse_loss(z_pred[:, :-1], z[:, 1:])
            sig_loss = sigreg(z.reshape(-1, z.shape[-1]))
            loss = pred_loss + lambd * sig_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_pred += pred_loss.item()
            total_sig += sig_loss.item()
            n += 1

        scheduler.step()
        avg_pred = total_pred / max(n, 1)
        avg_sig = total_sig / max(n, 1)
        avg_total = avg_pred + lambd * avg_sig

        if avg_total < best_loss:
            best_loss = avg_total

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{epochs} | pred={avg_pred:.6f} | "
                  f"sigreg={avg_sig:.6f} | total={avg_total:.6f}")


# ============================================================
# Evaluation: rollout quality
# ============================================================

def evaluate_rollout(encoder, predictor, dataset, n_rollouts=100, horizon=10):
    """Evaluate world model accuracy: predict H steps, measure error."""
    encoder.eval()
    predictor.eval()

    errors = []
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (obs, actions, length) in enumerate(loader):
        if i >= n_rollouts:
            break
        L = int(length[0])
        if L < horizon + 1:
            continue

        actions_norm = actions / 256.0 - 1.0

        with torch.no_grad():
            z_true = encoder(obs)  # (1, T, D)

            # Start from first state, predict forward
            z = z_true[:, 0:1, :]  # (1, 1, D)
            for t in range(min(horizon, L - 1)):
                a = actions_norm[:, t:t+1, :]  # (1, 1, A)
                z_next = predictor(z[:, -1:, :], a)
                z = torch.cat([z, z_next], dim=1)

            # Compare final predicted z to true z
            t_end = min(horizon, L - 1)
            pred_final = z[:, -1]
            true_final = z_true[:, t_end]
            error = (pred_final - true_final).pow(2).sum().sqrt().item()
            errors.append(error)

    if errors:
        print(f"  Rollout MSE (H={horizon}): mean={np.mean(errors):.4f}, "
              f"std={np.std(errors):.4f}, median={np.median(errors):.4f}")
    return errors


def evaluate_with_env(encoder, predictor, n_episodes=20):
    """Evaluate using CEM planner in the live Push-T environment."""
    try:
        import gymnasium as gym
        import gym_pusht
    except ImportError:
        print("  gym-pusht not available, skipping live eval")
        return []

    from planner import CEMPlanner

    planner = CEMPlanner(
        encoder, predictor,
        horizon=10, n_samples=128, n_iterations=3,
        top_k=16, n_actions=ACTION_DIM, discrete=False,
    )

    coverages = []
    for ep in range(n_episodes):
        env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
        obs, _ = env.reset(seed=ep + 1000)
        done = False
        steps = 0

        while not done and steps < 300:
            # Planner expects raw obs; normalize action output from [-1,1] to [0,512]
            action_norm = planner.plan(obs, cost_fn=_pusht_cost_fn(encoder, obs))
            action = (action_norm + 1.0) * 256.0
            action = np.clip(action, 0, 512).astype(np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        coverages.append(float(info.get("coverage", 0)))
        planner.reset()

        if (ep + 1) % 5 == 0:
            print(f"  [{ep+1}/{n_episodes}] avg coverage={np.mean(coverages):.3f}")

    return coverages


def _pusht_cost_fn(encoder, current_obs):
    """Cost function for Push-T: minimize distance between predicted and goal state."""
    def cost_fn(z_traj):
        # z_traj: (N, H+1, D)
        # Push-T doesn't have explicit goals in our random collection
        # Use a simple proxy: reward states far from current position
        z_start = z_traj[:, 0]
        z_end = z_traj[:, -1]
        # Maximize change in latent space (agent should be moving the block)
        change = (z_end - z_start).pow(2).sum(dim=-1)
        return -change  # lower is better
    return cost_fn


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LeWM for Push-T")
    parser.add_argument("--phase", default="all",
                        choices=["collect", "train", "eval", "all"])
    parser.add_argument("--data", default=None,
                        help="Path to pre-collected data (HDF5 or JSON)")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    obs_dim = STATE_DIM if not args.data or not args.data.endswith(".h5") else 2

    model_path = Path("results/lewm_pusht.pt")

    # Phase 1: Collect
    data_path = args.data
    if args.phase in ("collect", "all") and data_path is None:
        print(f"\n{'='*60}")
        print(f"Phase 1: Collecting {args.episodes} Push-T episodes")
        print(f"{'='*60}")
        from collect_distributed import collect_local, save_hdf5, save_json
        trajectories = collect_local("pusht", args.episodes)
        data_path = f"results/pusht_{args.episodes}"
        save_hdf5(trajectories, f"{data_path}.h5")
        # Also save JSON for richer state
        save_json(trajectories, f"{data_path}.json")
        data_path = f"{data_path}.json"  # Use JSON for full state

    if data_path is None:
        data_path = "results/pusht_500.h5"

    # Load dataset
    if data_path.endswith(".json"):
        dataset = PushTJSONDataset(data_path, seq_len=args.seq_len)
        obs_dim = dataset.samples[0]["obs"].shape[-1]  # detect from data
    else:
        dataset = PushTDataset(data_path, seq_len=args.seq_len)
        obs_dim = dataset.samples[0]["obs"].shape[-1]

    # Create models
    encoder = Encoder(obs_dim=obs_dim, embed_dim=args.embed_dim, hidden_dim=HIDDEN_DIM)
    predictor = Predictor(
        embed_dim=args.embed_dim, action_dim=ACTION_DIM,
        hidden_dim=args.embed_dim, n_layers=4, n_heads=4,
        discrete_actions=False, max_seq_len=args.seq_len,
    )
    sigreg = SIGReg(embed_dim=args.embed_dim)

    enc_p = sum(p.numel() for p in encoder.parameters())
    pred_p = sum(p.numel() for p in predictor.parameters())
    print(f"LeWM Push-T: {enc_p + pred_p:,} params (enc={enc_p:,} pred={pred_p:,})")

    # Phase 2: Train
    if args.phase in ("train", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 2: Training LeWM ({args.epochs} epochs, λ={args.lambd})")
        print(f"{'='*60}")
        train_lewm(encoder, predictor, sigreg, dataset,
                    epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size, lambd=args.lambd)
        model_path.parent.mkdir(exist_ok=True)
        torch.save({
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "config": {"obs_dim": obs_dim, "embed_dim": args.embed_dim,
                       "hidden_dim": HIDDEN_DIM},
        }, model_path)
        print(f"  Saved to {model_path}")
    elif model_path.exists():
        ckpt = torch.load(model_path, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        predictor.load_state_dict(ckpt["predictor"])

    # Phase 3: Evaluate
    if args.phase in ("eval", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 3: Evaluation")
        print(f"{'='*60}")

        print("\n  Rollout accuracy:")
        evaluate_rollout(encoder, predictor, dataset, n_rollouts=100, horizon=10)

        print("\n  Live environment (CEM planner):")
        coverages = evaluate_with_env(encoder, predictor, n_episodes=args.eval_episodes)
        if coverages:
            print(f"\n  Push-T Results:")
            print(f"    Avg coverage: {np.mean(coverages):.3f}")
            print(f"    Max coverage: {max(coverages):.3f}")

            # Random baseline
            import gymnasium as gym
            import gym_pusht
            rand_cov = []
            for i in range(args.eval_episodes):
                env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
                obs, _ = env.reset(seed=i + 2000)
                for _ in range(300):
                    obs, _, t, tr, info = env.step(env.action_space.sample())
                    if t or tr: break
                rand_cov.append(float(info.get("coverage", 0)))
            print(f"    Random baseline: {np.mean(rand_cov):.3f}")
            print(f"    Improvement: {np.mean(coverages) - np.mean(rand_cov):+.3f}")


if __name__ == "__main__":
    main()
