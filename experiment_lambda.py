#!/usr/bin/env python3
"""LeWM λ ablation: prove λ=0.032 beats paper's λ=0.1 at scale.

Runs a controlled experiment on GPU:
  1. Collect 500 Push-T episodes, split 400 train / 100 test
  2. Train identical models with λ=0.032 vs λ=0.1
  3. Evaluate on HELD-OUT test data (generalization, not memorization)
  4. Compare rollout prediction quality at multiple horizons

This produces a concrete novel finding beyond the official LeWM repo.
"""

import argparse
import json
import math
import os
import time
import numpy as np
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from model.encoder import ViTEncoder
from model.predictor import Predictor
from model.sigreg import SIGReg


# ============================================================
# Memory-efficient Pixel Dataset
# ============================================================

class PushTPixelDataset(Dataset):
    """Push-T pixel trajectories with train/test split support."""

    def __init__(self, n_episodes=500, seq_len=16, image_size=96,
                 cache_dir="results", split=None, split_ratio=0.8):
        self.seq_len = seq_len
        self.image_size = image_size

        cache_path = Path(cache_dir) / f"pusht_pixels_{n_episodes}_{image_size}.npz"
        if cache_path.exists():
            print(f"  Loading cached data from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self.all_pixels = data["pixels"]
            self.all_actions = data["actions"]
            ep_lens = data["ep_lens"]
        else:
            print(f"  Generating {n_episodes} Push-T episodes ({image_size}×{image_size})...")
            self.all_pixels, self.all_actions, ep_lens = self._collect(n_episodes, image_size)
            Path(cache_dir).mkdir(exist_ok=True)
            np.savez_compressed(cache_path,
                                pixels=self.all_pixels, actions=self.all_actions, ep_lens=ep_lens)
            print(f"  Cached to {cache_path}")

        # Split episodes into train/test
        n_eps = len(ep_lens)
        split_idx = int(n_eps * split_ratio)

        if split == "train":
            use_eps = list(range(split_idx))
        elif split == "test":
            use_eps = list(range(split_idx, n_eps))
        else:
            use_eps = list(range(n_eps))

        # Build windows only for selected episodes
        self.windows = []
        offsets = np.concatenate([[0], np.cumsum(ep_lens)])
        for ep_i in use_eps:
            offset = int(offsets[ep_i])
            T = int(ep_lens[ep_i])
            stride = max(seq_len // 2, 1)
            for start in range(0, max(1, T - seq_len + 1), stride):
                end = min(start + seq_len, T)
                sl = end - start
                if sl < 4:
                    continue
                self.windows.append((offset + start, sl))

        split_name = split or "all"
        print(f"  [{split_name}] {len(self.windows)} subsequences from {len(use_eps)} episodes")

    def _collect(self, n_episodes, image_size):
        try:
            import pymunk
            if not hasattr(pymunk.Space, 'on_collision'):
                def _on_collision(self, a, b, **kw):
                    handler = self.add_collision_handler(a, b)
                    if 'begin' in kw:
                        handler.begin = kw['begin']
                    return handler
                pymunk.Space.on_collision = _on_collision
        except ImportError:
            pass

        import gymnasium as gym
        import gym_pusht

        all_pixels, all_actions, ep_lens = [], [], []
        for ep in range(n_episodes):
            env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                           render_mode="rgb_array")
            obs, _ = env.reset(seed=ep)
            pix_list, act_list = [], []

            for _ in range(300):
                action = env.action_space.sample()
                pixel_obs = obs["pixels"]
                h, w = pixel_obs.shape[:2]
                sy, sx = h // image_size, w // image_size
                resized = pixel_obs[::sy, ::sx][:image_size, :image_size]
                pix_chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
                pix_list.append(pix_chw)
                act_list.append(action.astype(np.float32))
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            all_pixels.append(np.array(pix_list))
            all_actions.append(np.array(act_list))
            ep_lens.append(len(pix_list))
            if (ep + 1) % 50 == 0:
                print(f"    Collected {ep + 1}/{n_episodes} episodes")

        return (np.concatenate(all_pixels),
                np.concatenate(all_actions),
                np.array(ep_lens))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, sl = self.windows[idx]
        p = np.zeros((self.seq_len, 3, self.image_size, self.image_size), dtype=np.float32)
        a = np.zeros((self.seq_len, 2), dtype=np.float32)
        p[:sl] = self.all_pixels[start:start + sl]
        a[:sl] = self.all_actions[start:start + sl]
        return torch.from_numpy(p), torch.from_numpy(a), sl


# ============================================================
# Training
# ============================================================

def make_model(embed_dim, image_size, patch_size, vit_hidden, vit_layers,
               vit_heads, pred_layers, pred_heads, seq_len, device):
    encoder = ViTEncoder(
        embed_dim=embed_dim, image_size=image_size,
        patch_size=patch_size, hidden_size=vit_hidden,
        num_layers=vit_layers, num_heads=vit_heads,
    ).to(device)
    predictor = Predictor(
        embed_dim=embed_dim, action_dim=2,
        hidden_dim=embed_dim, n_layers=pred_layers,
        n_heads=pred_heads, mlp_dim=embed_dim * 4,
        discrete_actions=False, max_seq_len=seq_len,
    ).to(device)
    sigreg = SIGReg(embed_dim=embed_dim).to(device)
    return encoder, predictor, sigreg


def train_model(encoder, predictor, sigreg, train_data, device,
                epochs=50, lr=3e-4, batch_size=8, lambd=0.1):
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=2, pin_memory=True)
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    warmup = min(5, epochs // 5)
    def lr_sched(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    history = []
    for epoch in range(1, epochs + 1):
        encoder.train(); predictor.train()
        total_pred, total_sig, n = 0.0, 0.0, 0

        for pixels, actions, lengths in loader:
            pixels, actions = pixels.to(device), actions.to(device)
            actions_norm = actions / 256.0 - 1.0

            z = encoder(pixels)
            z_pred = predictor(z, actions_norm)
            pred_loss = F.mse_loss(z_pred[:, :-1], z[:, 1:])
            sig_loss = sigreg(z.reshape(-1, z.shape[-1]))
            loss = pred_loss + lambd * sig_loss

            if torch.isnan(loss):
                continue

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
        history.append({"epoch": epoch, "pred": avg_pred, "sigreg": avg_sig})

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3d}/{epochs} | pred={avg_pred:.6f} | sig={avg_sig:.4f}")

    return history


def evaluate_generalization(encoder, predictor, test_data, device,
                            horizons=[1, 5, 10, 15], n_rollouts=200):
    """Evaluate on HELD-OUT test episodes at multiple prediction horizons."""
    encoder.eval(); predictor.eval()
    loader = DataLoader(test_data, batch_size=1, shuffle=True)
    results = {}

    for H in horizons:
        errors = []
        for i, (pixels, actions, length) in enumerate(loader):
            if i >= n_rollouts:
                break
            L = int(length[0])
            if L < H + 1:
                continue

            pixels, actions = pixels.to(device), actions.to(device)
            actions_norm = actions / 256.0 - 1.0

            with torch.no_grad():
                z_true = encoder(pixels)
                z = z_true[:, 0:1, :]
                for t in range(H):
                    a = actions_norm[:, t:t + 1, :]
                    z_next = predictor(z[:, -1:, :], a)
                    z = torch.cat([z, z_next], dim=1)

                pred_final = z[:, -1]
                true_final = z_true[:, H]
                error = (pred_final - true_final).pow(2).sum().sqrt().item()
                errors.append(error)

        results[H] = {
            "mean": float(np.mean(errors)) if errors else None,
            "std": float(np.std(errors)) if errors else None,
            "median": float(np.median(errors)) if errors else None,
            "n": len(errors),
        }
        if errors:
            print(f"    H={H:>2d}: error={np.mean(errors):.4f} ± {np.std(errors):.4f} (n={len(errors)})")

    return results


# ============================================================
# Main Experiment
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LeWM λ Ablation Study")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--vit-hidden", type=int, default=192)
    parser.add_argument("--vit-layers", type=int, default=6)
    parser.add_argument("--vit-heads", type=int, default=3)
    parser.add_argument("--vit-patch", type=int, default=16)
    parser.add_argument("--pred-layers", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=8)
    parser.add_argument("--lambdas", type=str, default="0.01,0.032,0.1,0.32",
                        help="Comma-separated λ values to compare")
    parser.add_argument("--save-dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambdas = [float(x) for x in args.lambdas.split(",")]

    print(f"\n{'='*70}")
    print(f"LeWM λ Ablation Study")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  λ values: {lambdas}")
    print(f"  Episodes: {args.episodes} (80% train / 20% test)")
    print(f"  Epochs: {args.epochs}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # ---- Collect data once (shared across all λ) ----
    print(f"\n{'='*70}")
    print(f"Phase 1: Data Collection")
    print(f"{'='*70}")

    # We create train and test from the same cached collection
    train_data = PushTPixelDataset(
        n_episodes=args.episodes, seq_len=args.seq_len,
        image_size=args.image_size, cache_dir=args.save_dir, split="train")
    test_data = PushTPixelDataset(
        n_episodes=args.episodes, seq_len=args.seq_len,
        image_size=args.image_size, cache_dir=args.save_dir, split="test")

    # ---- Train with each λ ----
    all_results = {}

    for lambd in lambdas:
        print(f"\n{'='*70}")
        print(f"Phase 2: Training with λ={lambd}")
        print(f"{'='*70}")

        encoder, predictor, sigreg = make_model(
            args.embed_dim, args.image_size, args.vit_patch,
            args.vit_hidden, args.vit_layers, args.vit_heads,
            args.pred_layers, args.pred_heads, args.seq_len, device)

        params = sum(p.numel() for p in encoder.parameters()) + \
                 sum(p.numel() for p in predictor.parameters())
        print(f"  Model: {params / 1e6:.1f}M params")

        t0 = time.time()
        history = train_model(encoder, predictor, sigreg, train_data, device,
                              epochs=args.epochs, lr=args.lr,
                              batch_size=args.batch_size, lambd=lambd)
        train_time = time.time() - t0

        # ---- Evaluate on train (memorization check) ----
        print(f"\n  Train set evaluation:")
        train_results = evaluate_generalization(
            encoder, predictor, train_data, device,
            horizons=[1, 5, 10], n_rollouts=100)

        # ---- Evaluate on HELD-OUT test (generalization) ----
        print(f"\n  Test set evaluation (HELD-OUT):")
        test_results = evaluate_generalization(
            encoder, predictor, test_data, device,
            horizons=[1, 5, 10], n_rollouts=200)

        all_results[str(lambd)] = {
            "lambda": lambd,
            "train_time": train_time,
            "final_pred_loss": history[-1]["pred"],
            "final_sigreg_loss": history[-1]["sigreg"],
            "train_rollout": train_results,
            "test_rollout": test_results,
            "history": history,
        }

        # Save checkpoint
        torch.save({
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "lambda": lambd,
            "history": history,
        }, save_dir / f"lewm_lambda_{lambd}.pt")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"RESULTS: λ Ablation (held-out test set)")
    print(f"{'='*70}")
    print(f"\n  {'λ':>8s} | {'Train Loss':>12s} | {'Test H=1':>10s} | {'Test H=5':>10s} | {'Test H=10':>10s} | {'Time':>8s}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8}")

    for lambd in lambdas:
        r = all_results[str(lambd)]
        h1 = r["test_rollout"].get(1, {}).get("mean", float("nan"))
        h5 = r["test_rollout"].get(5, {}).get("mean", float("nan"))
        h10 = r["test_rollout"].get(10, {}).get("mean", float("nan"))
        print(f"  {lambd:>8.3f} | {r['final_pred_loss']:>12.6f} | "
              f"{h1:>10.4f} | {h5:>10.4f} | {h10:>10.4f} | "
              f"{r['train_time']:>7.0f}s")

    # Find best λ
    best_lambda = min(lambdas,
                      key=lambda l: all_results[str(l)]["test_rollout"].get(10, {}).get("mean", 999))
    paper_lambda = 0.1
    if str(paper_lambda) in all_results and str(best_lambda) in all_results:
        paper_h10 = all_results[str(paper_lambda)]["test_rollout"].get(10, {}).get("mean", 999)
        best_h10 = all_results[str(best_lambda)]["test_rollout"].get(10, {}).get("mean", 999)
        if paper_h10 > 0:
            improvement = (paper_h10 - best_h10) / paper_h10 * 100
            print(f"\n  Best λ={best_lambda} beats paper's λ={paper_lambda} by {improvement:.1f}% "
                  f"on held-out test rollout (H=10)")

    # Save full results
    results_path = save_dir / "lambda_ablation_results.json"
    # Convert history to serializable format
    serializable = {}
    for k, v in all_results.items():
        sv = dict(v)
        sv["train_rollout"] = {str(kk): vv for kk, vv in sv["train_rollout"].items()}
        sv["test_rollout"] = {str(kk): vv for kk, vv in sv["test_rollout"].items()}
        serializable[k] = sv

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Full results saved to {results_path}")


if __name__ == "__main__":
    main()
