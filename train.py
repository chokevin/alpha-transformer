#!/usr/bin/env python3
"""Decision Transformer training pipeline for commodity trading.

Three-phase training:
  1. Collect trajectories from the commodity environment (diverse heuristics)
  2. Supervised fine-tuning on top trajectories conditioned on returns (SFT)
  3. RL fine-tuning with PPO-style clipping against live environment

Usage:
    python train.py --phase all                    # full pipeline
    python train.py --phase collect --episodes 1000  # just collect
    python train.py --phase rl --episodes 5000     # RL only (after SFT)
    python train.py --config configs/default.yaml  # load from config
"""

import argparse
import json
import math
import time
import numpy as np
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.transformer import TradingTransformer
from model.policy import build_context, run_episode, save_checkpoint, load_checkpoint
from envs.commodities import CommodityTradingEnv
from envs.features import STATE_DIM


# ============================================================
# Dataset: Trading trajectories → sequence samples
# ============================================================

class TradingTrajectoryDataset(Dataset):
    """Convert trajectories into sliding-window samples for SFT.

    Each sample is a (states, actions, rtgs) sequence of max_seq_len steps.
    Uses overlapping windows with stride = max_seq_len // 2.
    """

    def __init__(self, trajectories, max_seq_len=60, gamma=0.99):
        self.max_seq_len = max_seq_len
        self.samples = []

        for traj in trajectories:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            rewards = np.array(traj["rewards"], dtype=np.float32)

            # Compute discounted return-to-go
            rtgs = np.zeros_like(rewards)
            rtg = 0
            for i in range(len(rewards) - 1, -1, -1):
                rtg = rewards[i] + gamma * rtg
                rtgs[i] = rtg

            # Sliding windows with overlap
            stride = max(max_seq_len // 2, 1)
            for start in range(0, max(1, len(states) - max_seq_len + 1), stride):
                end = min(start + max_seq_len, len(states))
                self.samples.append({
                    "states": states[start:end],
                    "actions": actions[start:end],
                    "rtgs": rtgs[start:end],
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
# Phase 1: Collect trajectories with diverse heuristics
# ============================================================

def _bollinger_action(obs, n_commodities=4):
    """Mean-reversion with long bias: buy dips, trim extended rallies."""
    actions = np.zeros(n_commodities)
    for i in range(n_commodities):
        base = i * 10  # 10 features per commodity
        bb_pos = obs[base + 8] if base + 8 < len(obs) else 0
        rsi = obs[base + 7] if base + 7 < len(obs) else 0
        # Long bias with mean-reversion timing
        # Base position is long, Bollinger guides sizing
        base_pos = 0.3
        timing = -bb_pos * 0.5 - rsi * 0.3  # buy low, reduce high
        actions[i] = np.clip(base_pos + timing, -0.5, 1.0)
    return actions


def _momentum_action(obs, n_commodities=4):
    """Trend-following with long bias: ride trends, scale by vol regime."""
    actions = np.zeros(n_commodities)
    for i in range(n_commodities):
        base = i * 10
        ret_5d = obs[base + 1] if base + 1 < len(obs) else 0
        ret_20d = obs[base + 2] if base + 2 < len(obs) else 0
        ma5_ratio = obs[base + 3] if base + 3 < len(obs) else 0
        vol_regime = obs[base + 6] if base + 6 < len(obs) else 0
        # Base long position + momentum tilt
        base_pos = 0.25
        signal = (ret_5d * 2 + ret_20d + ma5_ratio * 2) / 3
        vol_scale = max(0.3, 1.0 - vol_regime * 0.3)
        actions[i] = np.clip(base_pos + signal * vol_scale, -0.3, 1.0)
    return actions


def _risk_parity_action(obs, n_commodities=4):
    """Risk parity: inverse volatility weighting, always long."""
    actions = np.zeros(n_commodities)
    inv_vols = []
    for i in range(n_commodities):
        base = i * 10
        vol = (obs[base + 5] + 1) / 2  # denormalize from [-1,1] to [0,1]
        inv_vols.append(1.0 / (vol + 0.1))
    total = sum(inv_vols)
    for i in range(n_commodities):
        weight = inv_vols[i] / total
        actions[i] = np.clip(weight * 2.5, 0.1, 0.8)  # always long
    return actions


def _long_bias_action(obs, n_commodities=4):
    """Simple long bias with mild volatility scaling."""
    actions = np.zeros(n_commodities)
    for i in range(n_commodities):
        base = i * 10
        vol_regime = obs[base + 6] if base + 6 < len(obs) else 0
        # Reduce position in high vol, increase in low vol
        size = 0.4 - vol_regime * 0.15
        actions[i] = np.clip(size, 0.1, 0.7)
    return actions


def collect_trajectories(n_episodes, env_class, min_return_percentile=80):
    """Collect trajectories using diverse heuristics, filter top performers."""
    trajectories = []
    heuristics = [_bollinger_action, _momentum_action, _risk_parity_action, _long_bias_action]

    for ep in range(n_episodes):
        env = env_class(n_days=252)
        obs, _ = env.reset()
        states, actions, rewards = [], [], []

        # Select heuristic: 10% random, 90% from heuristic pool
        if np.random.random() < 0.1:
            policy_fn = lambda o: np.random.uniform(-0.1, 0.5, size=4)  # even random is long-biased
        else:
            policy_fn = heuristics[ep % len(heuristics)]

        done = False
        while not done:
            states.append(obs.tolist())
            action = policy_fn(obs)
            # Add small noise for diversity
            action = np.clip(action + np.random.normal(0, 0.05, size=4), -1, 1)
            actions.append(action.tolist())
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            done = terminated or truncated

        trajectories.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "total_return": info.get("total_return", 0),
            "sharpe": info.get("sharpe", 0),
        })

        if (ep + 1) % 100 == 0:
            recent = trajectories[-100:]
            returns = [t["total_return"] for t in recent]
            sharpes = [t["sharpe"] for t in recent]
            print(f"  Collected {ep+1}/{n_episodes} | "
                  f"avg return: {np.mean(returns):+.2f}% | "
                  f"avg sharpe: {np.mean(sharpes):.3f}")

    # Filter top trajectories
    returns = [t["total_return"] for t in trajectories]
    threshold = np.percentile(returns, min_return_percentile)
    filtered = [t for t in trajectories if t["total_return"] >= threshold]
    print(f"  Filtered: {len(filtered)}/{len(trajectories)} trajectories "
          f"(top {100-min_return_percentile}%, threshold={threshold:+.2f}%)")

    return filtered


# ============================================================
# Phase 2: Supervised fine-tuning
# ============================================================

def train_supervised(model, trajectories, epochs=50, lr=1e-4, batch_size=32):
    """Train transformer to predict actions from (state, return-to-go) sequences.

    Uses MSE loss + action magnitude regularization to prevent collapsing
    to near-zero actions (a common failure mode when training data has
    diverse action distributions).
    """
    dataset = TradingTrajectoryDataset(trajectories, max_seq_len=model.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine decay
    warmup_epochs = min(5, epochs // 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Compute target action magnitude from data for calibration
    all_actions = []
    for traj in trajectories:
        all_actions.extend(traj["actions"])
    target_mag = np.mean(np.abs(all_actions))
    print(f"  Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")
    print(f"  Target action magnitude: {target_mag:.3f}")

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_mag = 0
        n_batches = 0

        for states, actions, rtgs in loader:
            pred_actions = model(states, actions, rtgs)

            # MSE on action prediction
            mse_loss = F.mse_loss(pred_actions, actions)

            # Action magnitude regularization: penalize if predicted actions
            # are much smaller than target data magnitude
            pred_mag = pred_actions.abs().mean()
            mag_penalty = F.relu(target_mag * 0.5 - pred_mag)  # activate if pred < 50% of target

            loss = mse_loss + mag_penalty * 0.5

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += mse_loss.item()
            total_mag += pred_mag.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        avg_mag = total_mag / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{epochs} | loss={avg_loss:.6f} | "
                  f"|act|={avg_mag:.3f} | lr={scheduler.get_last_lr()[0]:.6f}")

    return model


# ============================================================
# Phase 3: RL fine-tuning — collect then re-score
# ============================================================

def rl_finetune(model, env_class, episodes=2000, batch_size=20,
                lr=1e-4, target_return=15.0, eval_interval=200,
                entropy_coef=0.005, clip_eps=0.2):
    """Fine-tune transformer with policy gradient.

    Efficient approach:
    1. Collect batch of episodes with no_grad (fast)
    2. Re-score a sample of (state, action) pairs with grad
    3. Compute policy gradient and update

    Uses annealed exploration noise schedule.
    """
    model.log_std.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    n_updates = episodes // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_updates)
    history = []
    best_sharpe = -float("inf")
    best_state = None

    for ep in range(0, episodes, batch_size):
        # Anneal exploration: 0.3 → 0.05
        progress = ep / max(episodes - batch_size, 1)
        current_std = 0.3 * (1 - progress) + 0.05 * progress
        model.log_std.data.fill_(math.log(current_std))

        # --- Step 1: Collect episodes with no_grad ---
        model.eval()
        collected = []  # list of (states, actions, rewards, ep_return)

        for _ in range(batch_size):
            env = env_class(n_days=252)
            obs, _ = env.reset()
            states_buf, actions_buf, rewards_ep = [], [], []
            done = False

            while not done:
                states_buf.append(obs)
                s, a, r = build_context(states_buf, actions_buf, rewards_ep,
                                        model, target_return)
                with torch.no_grad():
                    action = model.get_action(s, a, r, deterministic=False)
                action = action.squeeze(0).numpy()
                actions_buf.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards_ep.append(reward)
                done = terminated or truncated

            # Compute discounted returns per timestep
            G = 0
            returns = []
            for rew in reversed(rewards_ep):
                G = rew + 0.99 * G
                returns.insert(0, G)

            collected.append({
                "states": states_buf,
                "actions": actions_buf,
                "returns": returns,
                "ep_return": info.get("total_return", 0),
            })

        # --- Step 2: Re-score sampled timesteps with grad ---
        model.train()
        all_log_probs = []
        all_advantages = []

        # Flatten all returns for normalization
        all_returns = []
        for ep_data in collected:
            all_returns.extend(ep_data["returns"])
        ret_mean = np.mean(all_returns)
        ret_std = np.std(all_returns) + 1e-8

        for ep_data in collected:
            states_list = ep_data["states"]
            actions_list = ep_data["actions"]
            returns_list = ep_data["returns"]
            n_steps = len(states_list)

            # Sample a subset of timesteps (every 10th) for efficiency
            sample_indices = list(range(0, n_steps, 10))
            if not sample_indices:
                sample_indices = [0]

            for t in sample_indices:
                # Rebuild context up to timestep t
                seq_len = min(t + 1, model.max_seq_len)
                start = max(0, t + 1 - seq_len)

                s_arr = np.array(states_list[start:t + 1], dtype=np.float32)
                s = torch.tensor(s_arr).unsqueeze(0)

                if t > 0:
                    a_start = max(0, t - seq_len)
                    a_arr = np.array(actions_list[a_start:t], dtype=np.float32)
                    a = torch.tensor(a_arr).unsqueeze(0)
                    if a.shape[1] < s.shape[1]:
                        pad = torch.zeros(1, s.shape[1] - a.shape[1], model.action_dim)
                        a = torch.cat([pad, a], dim=1)
                else:
                    a = torch.zeros(1, seq_len, model.action_dim)

                rtg = (target_return - sum(ep_data["returns"][:t])) / 100.0
                r = torch.full((1, seq_len, 1), rtg, dtype=torch.float32)

                # Forward with grad
                mu = model.forward(s, a, r)[:, -1]
                dist = torch.distributions.Normal(mu, current_std)

                action_taken = torch.tensor(actions_list[t], dtype=torch.float32).unsqueeze(0)
                log_prob = dist.log_prob(action_taken).sum()

                advantage = (returns_list[t] - ret_mean) / ret_std
                advantage = np.clip(advantage, -clip_eps * 10, clip_eps * 10)

                all_log_probs.append(log_prob)
                all_advantages.append(advantage)

        if not all_log_probs:
            continue

        # --- Step 3: Policy gradient update ---
        log_probs_t = torch.stack(all_log_probs)
        advantages_t = torch.tensor(all_advantages, dtype=torch.float32)

        loss = -(log_probs_t * advantages_t).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        # Evaluate
        if (ep + batch_size) % eval_interval == 0 or ep == 0:
            eval_returns, eval_sharpes = [], []
            eval_action_mags = []
            model.eval()
            for _ in range(20):
                env = env_class(n_days=252)
                info, traj = run_episode(model, env, target_return=target_return,
                                         deterministic=True)
                eval_returns.append(info.get("total_return", 0))
                eval_sharpes.append(info.get("sharpe", 0))
                eval_action_mags.append(np.mean(np.abs(traj["actions"])))

            avg_ret = np.mean(eval_returns)
            avg_sharpe = np.mean(eval_sharpes)
            avg_mag = np.mean(eval_action_mags)

            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                marker = " ★"
            else:
                marker = ""

            bar = "█" * int(np.clip((avg_ret + 20) * 1.25, 0, 50))
            bar += "░" * (50 - len(bar))
            print(f"  Ep {ep+batch_size:>5d} | ret={avg_ret:>+7.2f}% | "
                  f"sharpe={avg_sharpe:>+.3f} | |act|={avg_mag:.3f} | "
                  f"σ={current_std:.3f} | {bar}{marker}")

            history.append({
                "episode": ep + batch_size,
                "avg_return": round(avg_ret, 2),
                "avg_sharpe": round(avg_sharpe, 3),
                "loss": round(loss.item(), 4),
            })

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (Sharpe={best_sharpe:.3f})")

    return model, history


# ============================================================
# Config loading
# ============================================================

def load_config(path):
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_args_config(args, config):
    """Merge CLI args with config file (CLI takes precedence for explicit args)."""
    if config is None:
        return args

    # Map config keys to args
    mapping = {
        ("model", "hidden_dim"): "hidden",
        ("model", "n_heads"): "heads",
        ("model", "n_layers"): "layers",
        ("model", "max_seq_len"): "seq_len",
        ("model", "dropout"): "dropout",
        ("collect", "episodes"): "episodes",
        ("collect", "min_return_percentile"): "min_return_percentile",
        ("sft", "epochs"): "epochs",
        ("sft", "lr"): "sft_lr",
        ("sft", "batch_size"): "sft_batch_size",
        ("rl", "episodes"): "rl_episodes",
        ("rl", "batch_size"): "rl_batch_size",
        ("rl", "lr"): "rl_lr",
        ("rl", "target_return"): "target_return",
        ("rl", "eval_interval"): "eval_interval",
        ("environment", "n_days"): "n_days",
        ("environment", "initial_cash"): "initial_cash",
        ("environment", "commission_bps"): "commission_bps",
    }

    for (section, key), attr in mapping.items():
        if section in config and key in config[section]:
            if not hasattr(args, attr) or getattr(args, attr) is None:
                setattr(args, attr, config[section][key])

    return args


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Decision Transformer for trading")
    parser.add_argument("--phase", default="all",
                        choices=["collect", "train", "rl", "all"])
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config file")

    # Model
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # Collection
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--min-return-percentile", type=int, default=None)

    # SFT
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sft-lr", type=float, default=None)
    parser.add_argument("--sft-batch-size", type=int, default=None)

    # RL
    parser.add_argument("--rl-episodes", type=int, default=None)
    parser.add_argument("--rl-batch-size", type=int, default=None)
    parser.add_argument("--rl-lr", type=float, default=None)
    parser.add_argument("--target-return", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)

    args = parser.parse_args()

    # Load config
    config = None
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f"Loaded config: {args.config}")

    args = merge_args_config(args, config)

    # Apply defaults for anything still None
    defaults = {
        "hidden": 128, "layers": 4, "heads": 4, "seq_len": 60, "dropout": 0.1,
        "episodes": 2000, "min_return_percentile": 80,
        "epochs": 100, "sft_lr": 3e-4, "sft_batch_size": 64,
        "rl_episodes": 2000, "rl_batch_size": 20, "rl_lr": 1e-4,
        "target_return": 15.0, "eval_interval": 200,
        "n_days": 252, "initial_cash": 100000, "commission_bps": 2,
    }
    for k, v in defaults.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)

    model = TradingTransformer(
        state_dim=STATE_DIM, action_dim=4,
        hidden_dim=args.hidden, n_heads=args.heads,
        n_layers=args.layers, max_seq_len=args.seq_len,
        dropout=args.dropout,
    )
    param_count = model.param_count()
    print(f"Model: {param_count:,} parameters | state_dim={STATE_DIM} | "
          f"hidden={args.hidden} | layers={args.layers}")

    traj_path = Path("results/trajectories.json")
    model_path = Path("results/trading_transformer.pt")

    # Phase 1: Collect
    if args.phase in ("collect", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 1: Collecting {args.episodes} trajectories")
        print(f"{'='*60}")
        trajectories = collect_trajectories(
            args.episodes, CommodityTradingEnv,
            min_return_percentile=args.min_return_percentile,
        )
        traj_path.parent.mkdir(exist_ok=True)
        with open(traj_path, "w") as f:
            json.dump(trajectories, f)
        print(f"  Saved {len(trajectories)} trajectories to {traj_path}")
    else:
        if traj_path.exists():
            with open(traj_path) as f:
                trajectories = json.load(f)
            print(f"  Loaded {len(trajectories)} trajectories from {traj_path}")
        else:
            print(f"  No trajectories found at {traj_path}. Run --phase collect first.")
            return

    # Phase 2: Supervised training
    if args.phase in ("train", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 2: Supervised training ({args.epochs} epochs)")
        print(f"{'='*60}")
        model = train_supervised(
            model, trajectories, epochs=args.epochs,
            lr=args.sft_lr, batch_size=args.sft_batch_size,
        )
        model_path.parent.mkdir(exist_ok=True)
        save_checkpoint(model, model_path, metadata={"phase": "sft"})
        print(f"  Saved to {model_path}")
    elif model_path.exists():
        model, _ = load_checkpoint(model_path, TradingTransformer)
        print(f"  Loaded model from {model_path}")

    # Phase 3: RL fine-tuning
    if args.phase in ("rl", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 3: RL fine-tuning ({args.rl_episodes} episodes)")
        print(f"{'='*60}")
        model, history = rl_finetune(
            model, CommodityTradingEnv,
            episodes=args.rl_episodes, batch_size=args.rl_batch_size,
            lr=args.rl_lr, target_return=args.target_return,
            eval_interval=args.eval_interval,
        )
        save_checkpoint(model, model_path, metadata={
            "phase": "rl", "history": history,
        })
        print(f"  Saved to {model_path}")

        history_path = Path(f"results/rl_history_{int(time.time())}.json")
        with open(history_path, "w") as f:
            json.dump({"history": history}, f, indent=2)

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Final Evaluation (50 episodes)")
    print(f"{'='*60}")
    model.eval()

    eval_returns, eval_sharpes = [], []
    for _ in range(50):
        env = CommodityTradingEnv(n_days=252)
        info, _ = run_episode(model, env, target_return=args.target_return,
                              deterministic=True)
        eval_returns.append(info.get("total_return", 0))
        eval_sharpes.append(info.get("sharpe", 0))

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

    agent_sharpe = np.mean(eval_sharpes)
    alpha = np.mean(eval_returns) - np.mean(bh_returns)

    print(f"\n  {'Model':<20} {'Return':>8} {'Std':>8} {'Win%':>6} {'Sharpe':>8}")
    print(f"  {'─'*52}")
    print(f"  {'Transformer':<20} {np.mean(eval_returns):>+7.2f}% "
          f"{np.std(eval_returns):>7.2f}% "
          f"{sum(1 for r in eval_returns if r > 0)/len(eval_returns)*100:>5.0f}% "
          f"{agent_sharpe:>+7.3f}")
    print(f"  {'Buy & Hold':<20} {np.mean(bh_returns):>+7.2f}% "
          f"{np.std(bh_returns):>7.2f}%")
    print(f"  {'─'*52}")
    print(f"  {'Alpha':<20} {alpha:>+7.2f}%")


if __name__ == "__main__":
    main()
