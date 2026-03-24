#!/usr/bin/env python3
"""Train the market world model (LeWM-style).

Single phase: predict next latent state. Two losses. One hyperparameter.

Usage:
    python train.py                          # train on simulated data
    python train.py --data trajectories.npz  # train on collected data
    python train.py --epochs 200 --lr 1e-4
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from model.world_model import MarketWorldModel
from envs.commodities import CommodityTradingEnv


def collect_trajectories(n_episodes=500, n_days=252):
    """Collect trajectories from the commodity env using heuristic policies."""
    all_obs, all_actions = [], []

    for ep in range(n_episodes):
        env = CommodityTradingEnv(n_days=n_days)
        obs, _ = env.reset()
        ep_obs, ep_actions = [obs], []

        done = False
        while not done:
            # Mix of momentum + random exploration
            if np.random.random() < 0.3:
                action = np.random.uniform(-0.5, 0.5, size=4)
            else:
                momentum = np.array([obs[3], obs[11], obs[19], obs[27]])
                action = np.clip(momentum * 3 + np.random.randn(4) * 0.2, -1, 1)

            obs, _, terminated, truncated, info = env.step(action)
            ep_obs.append(obs)
            ep_actions.append(action)
            done = terminated or truncated

        all_obs.append(np.array(ep_obs, dtype=np.float32))
        all_actions.append(np.array(ep_actions, dtype=np.float32))

        if (ep + 1) % 100 == 0:
            print(f"  Collected {ep+1}/{n_episodes}")

    return all_obs, all_actions


def make_batches(all_obs, all_actions, seq_len=4, batch_size=128):
    """Create training batches: short subsequences of (obs, action)."""
    obs_seqs, act_seqs = [], []

    for obs, actions in zip(all_obs, all_actions):
        T = min(len(obs) - 1, len(actions))
        for start in range(0, T - seq_len, seq_len // 2):
            end = start + seq_len
            if end > T:
                break
            obs_seqs.append(obs[start:end + 1])  # T+1 obs for T actions
            act_seqs.append(actions[start:end])

    obs_arr = np.array(obs_seqs, dtype=np.float32)
    act_arr = np.array(act_seqs, dtype=np.float32)

    # Shuffle
    idx = np.random.permutation(len(obs_arr))
    obs_arr, act_arr = obs_arr[idx], act_arr[idx]

    # Yield batches
    batches = []
    for i in range(0, len(obs_arr) - batch_size, batch_size):
        batches.append((
            torch.tensor(obs_arr[i:i + batch_size]),
            torch.tensor(act_arr[i:i + batch_size]),
        ))
    return batches


def evaluate_world_model(model, n_episodes=50):
    """Evaluate by measuring prediction error on held-out episodes."""
    model.eval()
    errors = []

    for _ in range(n_episodes):
        env = CommodityTradingEnv(n_days=50)
        obs, _ = env.reset()
        obs_list, act_list = [obs], []

        done = False
        while not done:
            action = np.random.uniform(-0.3, 0.3, size=4)
            obs, _, terminated, truncated, _ = env.step(action)
            obs_list.append(obs)
            act_list.append(action)
            done = terminated or truncated

        obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32).unsqueeze(0)
        act_t = torch.tensor(np.array(act_list), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            z_all = model.encode(obs_t)
            z_pred = model.predict(z_all[:, :-1], act_t)
            z_true = z_all[:, 1:]
            error = (z_pred - z_true).pow(2).mean().item()
            errors.append(error)

    model.train()
    return np.mean(errors)


def main():
    parser = argparse.ArgumentParser(description="Train market world model")
    parser.add_argument("--episodes", type=int, default=500, help="Collection episodes")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4, help="Subsequence length (LeWM uses 4)")
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--sigreg-weight", type=float, default=0.09, help="λ — the ONE hyperparameter")
    parser.add_argument("--data", type=str, default=None, help="Path to pre-collected .npz")
    args = parser.parse_args()

    # Model
    model = MarketWorldModel(
        obs_dim=52, action_dim=4,
        embed_dim=args.embed_dim, hidden_dim=256,
        pred_hidden=args.embed_dim, n_layers=4, n_heads=4,
        max_seq_len=args.seq_len * 10,
    )
    print(f"Model: {model.param_count():,} parameters")

    # Data
    if args.data and Path(args.data).exists():
        print(f"\nLoading data from {args.data}")
        data = np.load(args.data)
        all_obs = [data["observations"][i, :data["lengths"][i]]
                    for i in range(len(data["lengths"]))]
        all_actions = [data["actions"][i, :data["lengths"][i]]
                       for i in range(len(data["lengths"]))]
    else:
        print(f"\nCollecting {args.episodes} trajectories...")
        all_obs, all_actions = collect_trajectories(args.episodes)

    batches = make_batches(all_obs, all_actions,
                           seq_len=args.seq_len, batch_size=args.batch_size)
    print(f"Training batches: {len(batches)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    print(f"\n{'='*60}")
    print(f"Training World Model (LeWM-style)")
    print(f"{'='*60}")
    print(f"  Params:       {model.param_count():,}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batches/epoch:{len(batches)}")
    print(f"  Seq length:   {args.seq_len}")
    print(f"  λ (SIGReg):   {args.sigreg_weight}")
    print(f"{'='*60}\n")

    start = time.time()
    history = []

    for epoch in range(1, args.epochs + 1):
        total_loss, total_pred, total_sig = 0, 0, 0
        n = 0

        np.random.shuffle(batches)
        for obs_batch, act_batch in batches:
            loss, pred_loss, sig_loss = model.training_step(
                obs_batch, act_batch, sigreg_weight=args.sigreg_weight
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pred += pred_loss.item()
            total_sig += sig_loss.item()
            n += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / n
            avg_pred = total_pred / n
            avg_sig = total_sig / n
            eval_error = evaluate_world_model(model)

            print(f"  Epoch {epoch:>3d} | loss={avg_loss:.6f} "
                  f"(pred={avg_pred:.6f} sig={avg_sig:.6f}) | "
                  f"eval_error={eval_error:.6f}")

            history.append({
                "epoch": epoch,
                "loss": round(avg_loss, 6),
                "pred_loss": round(avg_pred, 6),
                "sigreg_loss": round(avg_sig, 6),
                "eval_error": round(eval_error, 6),
            })

    elapsed = time.time() - start

    # Save
    Path("results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "results/world_model.pt")
    with open(f"results/training_{int(time.time())}.json", "w") as f:
        json.dump({"history": history, "elapsed": elapsed, "args": vars(args)}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Final loss:  {history[-1]['loss']}")
    print(f"  Pred loss:   {history[-1]['pred_loss']}")
    print(f"  SIGReg loss: {history[-1]['sigreg_loss']}")
    print(f"  Eval error:  {history[-1]['eval_error']}")
    print(f"  Saved: results/world_model.pt")


if __name__ == "__main__":
    main()
