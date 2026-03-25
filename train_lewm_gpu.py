#!/usr/bin/env python3
"""LeWM GPU training: ViT pixel world model on Push-T.

Runs the full pipeline on GPU:
  1. Generate Push-T pixel trajectories
  2. Train ViT encoder + Transformer predictor with SIGReg
  3. Evaluate rollout prediction quality

This is the GPU-scale version that demonstrates:
  - No NaN issues (CPU float32 caused ~17% NaN batches)
  - Faster training (T4: ~10x vs CPU)
  - ViT-Tiny architecture matching the LeWM paper

Usage:
    python train_lewm_gpu.py --epochs 50 --episodes 200 --lambd 0.032
"""

import argparse
import json
import math
import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.encoder import ViTEncoder, Encoder
from model.predictor import Predictor
from model.sigreg import SIGReg


# ============================================================
# Push-T Pixel Dataset
# ============================================================

class PushTPixelDataset(Dataset):
    """Generate and serve Push-T pixel trajectories (memory-efficient)."""

    def __init__(self, n_episodes=200, seq_len=16, image_size=96, cache_dir="results"):
        self.seq_len = seq_len
        self.image_size = image_size

        cache_path = Path(cache_dir) / f"pusht_pixels_{n_episodes}_{image_size}.npz"
        if cache_path.exists():
            print(f"  Loading cached pixel data from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self.all_pixels = data["pixels"]
            self.all_actions = data["actions"]
            ep_lens = data["ep_lens"]
        else:
            print(f"  Generating {n_episodes} Push-T episodes ({image_size}×{image_size} pixels)...")
            self.all_pixels, self.all_actions, ep_lens = self._collect(n_episodes, image_size)
            Path(cache_dir).mkdir(exist_ok=True)
            np.savez_compressed(cache_path,
                                pixels=self.all_pixels, actions=self.all_actions, ep_lens=ep_lens)
            print(f"  Cached to {cache_path}")

        # Build index of (global_start, length) for each window — no pixel copies
        self.windows = []
        offset = 0
        for ep_len in ep_lens:
            T = int(ep_len)
            stride = max(seq_len // 2, 1)
            for start in range(0, max(1, T - seq_len + 1), stride):
                end = min(start + seq_len, T)
                sl = end - start
                if sl < 4:
                    continue
                self.windows.append((offset + start, sl))
            offset += T

        print(f"  Dataset: {len(self.windows)} subsequences from {len(ep_lens)} episodes")
        mb = self.all_pixels.nbytes / 1e6
        print(f"  Pixel data: {mb:.0f} MB in memory")

    def _collect(self, n_episodes, image_size):
        """Collect Push-T pixel episodes."""
        # Monkeypatch pymunk for gym-pusht compatibility
        try:
            import pymunk
            if not hasattr(pymunk.Space, 'on_collision'):
                def _on_collision(self, collision_type_a, collision_type_b, **kwargs):
                    handler = self.add_collision_handler(collision_type_a, collision_type_b)
                    if 'begin' in kwargs:
                        handler.begin = kwargs['begin']
                    return handler
                pymunk.Space.on_collision = _on_collision
        except ImportError:
            pass

        import gymnasium as gym
        try:
            import gym_pusht
        except ImportError:
            print("  gym-pusht not available, generating synthetic data")
            return self._synthetic_data(n_episodes, image_size)

        all_pixels = []
        all_actions = []
        ep_lens = []

        for ep in range(n_episodes):
            env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                           render_mode="rgb_array")
            obs, _ = env.reset(seed=ep)
            pixels_list = []
            actions_list = []

            for step in range(300):
                action = env.action_space.sample()
                # Extract pixel observation and resize
                pixel_obs = obs["pixels"]  # (H, W, 3) uint8
                pixel_resized = self._resize(pixel_obs, image_size)
                # Normalize to [0, 1] and CHW format
                pixel_chw = pixel_resized.transpose(2, 0, 1).astype(np.float32) / 255.0

                pixels_list.append(pixel_chw)
                actions_list.append(action.astype(np.float32))

                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            all_pixels.append(np.array(pixels_list))
            all_actions.append(np.array(actions_list))
            ep_lens.append(len(pixels_list))

            if (ep + 1) % 50 == 0:
                print(f"    Collected {ep + 1}/{n_episodes} episodes")

        return (np.concatenate(all_pixels),
                np.concatenate(all_actions),
                np.array(ep_lens))

    def _synthetic_data(self, n_episodes, image_size):
        """Fallback: generate synthetic moving-dot pixel data."""
        all_pixels = []
        all_actions = []
        ep_lens = []

        for ep in range(n_episodes):
            T = 100
            pos = np.array([image_size / 2, image_size / 2], dtype=np.float32)
            pixels_list = []
            actions_list = []

            for t in range(T):
                action = np.random.randn(2).astype(np.float32) * 5
                # Draw dot at position
                img = np.zeros((3, image_size, image_size), dtype=np.float32)
                x, y = int(np.clip(pos[0], 2, image_size - 3)), int(np.clip(pos[1], 2, image_size - 3))
                img[:, y-2:y+3, x-2:x+3] = 1.0

                pixels_list.append(img)
                actions_list.append(action)
                pos = np.clip(pos + action, 0, image_size - 1)

            all_pixels.append(np.array(pixels_list))
            all_actions.append(np.array(actions_list))
            ep_lens.append(T)

        return (np.concatenate(all_pixels),
                np.concatenate(all_actions),
                np.array(ep_lens))

    def _resize(self, img, size):
        """Simple resize using stride-based downsampling."""
        h, w = img.shape[:2]
        if h == size and w == size:
            return img
        # Use numpy stride tricks for fast downsampling
        sy, sx = h // size, w // size
        return img[::sy, ::sx][:size, :size]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, sl = self.windows[idx]
        p = np.zeros((self.seq_len, 3, self.image_size, self.image_size), dtype=np.float32)
        a = np.zeros((self.seq_len, 2), dtype=np.float32)
        p[:sl] = self.all_pixels[start:start + sl]
        a[:sl] = self.all_actions[start:start + sl]
        return torch.from_numpy(p), torch.from_numpy(a), sl


class PushTStateDataset(Dataset):
    """State-based Push-T dataset for MLP encoder baseline."""

    def __init__(self, n_episodes=500, seq_len=20):
        self.seq_len = seq_len
        self.samples = []

        try:
            import pymunk
            if not hasattr(pymunk.Space, 'on_collision'):
                def _on_collision(self, collision_type_a, collision_type_b, **kwargs):
                    handler = self.add_collision_handler(collision_type_a, collision_type_b)
                    if 'begin' in kwargs:
                        handler.begin = kwargs['begin']
                    return handler
                pymunk.Space.on_collision = _on_collision
        except ImportError:
            pass

        import gymnasium as gym
        import gym_pusht

        print(f"  Collecting {n_episodes} state-based Push-T episodes...")
        for ep in range(n_episodes):
            env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
            obs, _ = env.reset(seed=ep)
            states, actions = [], []

            for _ in range(300):
                action = env.action_space.sample()
                states.append(obs.astype(np.float32))
                actions.append(action.astype(np.float32))
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            ep_states = np.array(states)
            ep_actions = np.array(actions)
            T = len(ep_states)
            stride = max(seq_len // 2, 1)
            for start in range(0, max(1, T - seq_len + 1), stride):
                end = min(start + seq_len, T)
                sl = end - start
                if sl < 4:
                    continue
                s = np.zeros((seq_len, ep_states.shape[1]), dtype=np.float32)
                a = np.zeros((seq_len, 2), dtype=np.float32)
                s[:sl] = ep_states[start:end]
                a[:sl] = ep_actions[start:end]
                self.samples.append({"obs": s, "actions": a, "length": sl})

            if (ep + 1) % 100 == 0:
                print(f"    Collected {ep + 1}/{n_episodes}")

        print(f"  Dataset: {len(self.samples)} subsequences")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.tensor(s["obs"]), torch.tensor(s["actions"]), s["length"]


# ============================================================
# Training
# ============================================================

def train_lewm(encoder, predictor, sigreg, dataset, device,
               epochs=50, lr=3e-4, batch_size=32, lambd=0.032,
               pixel_mode=False):
    """Train LeWM on GPU with full metrics tracking."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=2, pin_memory=True)

    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    warmup = min(5, epochs // 5)
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n  Training config:")
    print(f"    Device: {device}")
    print(f"    Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")
    print(f"    Loss: MSE + {lambd} × SIGReg")
    print(f"    Pixel mode: {pixel_mode}")

    history = {"pred_loss": [], "sigreg_loss": [], "total_loss": [],
               "nan_batches": [], "epoch_time": []}
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        encoder.train()
        predictor.train()
        total_pred, total_sig = 0.0, 0.0
        n_ok, n_nan = 0, 0
        t0 = time.time()

        for batch in loader:
            if pixel_mode:
                pixels, actions, lengths = batch
                pixels = pixels.to(device)
            else:
                obs, actions, lengths = batch
                obs = obs.to(device)
            actions = actions.to(device)

            actions_norm = actions / 256.0 - 1.0

            if pixel_mode:
                z = encoder(pixels)
            else:
                z = encoder(obs)

            z_pred = predictor(z, actions_norm)

            pred_loss = F.mse_loss(z_pred[:, :-1], z[:, 1:])
            sig_loss = sigreg(z.reshape(-1, z.shape[-1]))
            loss = pred_loss + lambd * sig_loss

            if torch.isnan(loss):
                n_nan += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_pred += pred_loss.item()
            total_sig += sig_loss.item()
            n_ok += 1

        scheduler.step()
        dt = time.time() - t0
        avg_pred = total_pred / max(n_ok, 1)
        avg_sig = total_sig / max(n_ok, 1)
        avg_total = avg_pred + lambd * avg_sig
        nan_pct = 100 * n_nan / max(n_ok + n_nan, 1)

        history["pred_loss"].append(avg_pred)
        history["sigreg_loss"].append(avg_sig)
        history["total_loss"].append(avg_total)
        history["nan_batches"].append(nan_pct)
        history["epoch_time"].append(dt)

        if avg_total < best_loss:
            best_loss = avg_total

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{epochs} | pred={avg_pred:.6f} | "
                  f"sigreg={avg_sig:.4f} | time={dt:.1f}s | "
                  f"NaN={nan_pct:.0f}%")

    return history


def evaluate_rollout(encoder, predictor, dataset, device,
                     n_rollouts=100, horizon=10, pixel_mode=False):
    """Evaluate world model rollout prediction accuracy."""
    encoder.eval()
    predictor.eval()
    errors = []
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(loader):
        if i >= n_rollouts:
            break

        if pixel_mode:
            pixels, actions, length = batch
            pixels = pixels.to(device)
        else:
            obs, actions, length = batch
            obs = obs.to(device)
        actions = actions.to(device)

        L = int(length[0])
        if L < horizon + 1:
            continue

        actions_norm = actions / 256.0 - 1.0

        with torch.no_grad():
            if pixel_mode:
                z_true = encoder(pixels)
            else:
                z_true = encoder(obs)

            z = z_true[:, 0:1, :]
            for t in range(min(horizon, L - 1)):
                a = actions_norm[:, t:t + 1, :]
                z_next = predictor(z[:, -1:, :], a)
                z = torch.cat([z, z_next], dim=1)

            t_end = min(horizon, L - 1)
            pred_final = z[:, -1]
            true_final = z_true[:, t_end]
            error = (pred_final - true_final).pow(2).sum().sqrt().item()
            errors.append(error)

    if errors:
        print(f"  Rollout error (H={horizon}): mean={np.mean(errors):.4f}, "
              f"std={np.std(errors):.4f}, median={np.median(errors):.4f}")
    return errors


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LeWM GPU Training")
    parser.add_argument("--mode", default="pixel",
                        choices=["pixel", "state", "both"],
                        help="Training mode: pixel (ViT), state (MLP), or both")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lambd", type=float, default=0.032,
                        help="SIGReg weight (our sweep found 0.032 > paper's 0.1)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--vit-hidden", type=int, default=192,
                        help="ViT hidden dim (192 = ViT-Tiny)")
    parser.add_argument("--vit-layers", type=int, default=12)
    parser.add_argument("--vit-heads", type=int, default=3)
    parser.add_argument("--vit-patch", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--pred-layers", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"LeWM GPU Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Mode: {args.mode}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    results = {}

    # ---- PIXEL MODE (ViT) ----
    if args.mode in ("pixel", "both"):
        print(f"\n{'='*60}")
        print(f"ViT Pixel Training (image={args.image_size}, ViT-{args.vit_hidden})")
        print(f"{'='*60}")

        dataset = PushTPixelDataset(
            n_episodes=args.episodes, seq_len=args.seq_len,
            image_size=args.image_size, cache_dir=args.save_dir)

        encoder = ViTEncoder(
            embed_dim=args.embed_dim, image_size=args.image_size,
            patch_size=args.vit_patch, hidden_size=args.vit_hidden,
            num_layers=args.vit_layers, num_heads=args.vit_heads,
        ).to(device)

        predictor = Predictor(
            embed_dim=args.embed_dim, action_dim=2,
            hidden_dim=args.embed_dim, n_layers=args.pred_layers,
            n_heads=args.pred_heads, mlp_dim=args.embed_dim * 4,
            discrete_actions=False, max_seq_len=args.seq_len,
        ).to(device)

        sigreg = SIGReg(embed_dim=args.embed_dim).to(device)

        enc_p = sum(p.numel() for p in encoder.parameters())
        pred_p = sum(p.numel() for p in predictor.parameters())
        print(f"  Model: {(enc_p + pred_p) / 1e6:.1f}M params "
              f"(encoder={enc_p / 1e6:.1f}M, predictor={pred_p / 1e6:.1f}M)")

        t_start = time.time()
        history = train_lewm(
            encoder, predictor, sigreg, dataset, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            lambd=args.lambd, pixel_mode=True)
        train_time = time.time() - t_start

        print(f"\n  Evaluating rollout quality...")
        errors = evaluate_rollout(
            encoder, predictor, dataset, device,
            n_rollouts=100, horizon=10, pixel_mode=True)

        results["pixel"] = {
            "train_time": train_time,
            "final_pred_loss": history["pred_loss"][-1],
            "final_sigreg_loss": history["sigreg_loss"][-1],
            "avg_nan_pct": np.mean(history["nan_batches"]),
            "rollout_error_mean": np.mean(errors) if errors else None,
            "rollout_error_std": np.std(errors) if errors else None,
            "params": enc_p + pred_p,
            "avg_epoch_time": np.mean(history["epoch_time"]),
        }

        # Save model
        torch.save({
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "config": vars(args),
            "history": history,
        }, save_dir / "lewm_vit_pusht.pt")

    # ---- STATE MODE (MLP) ----
    if args.mode in ("state", "both"):
        print(f"\n{'='*60}")
        print(f"MLP State Training (baseline)")
        print(f"{'='*60}")

        dataset = PushTStateDataset(n_episodes=args.episodes, seq_len=args.seq_len)
        obs_dim = dataset.samples[0]["obs"].shape[-1]

        encoder = Encoder(
            obs_dim=obs_dim, embed_dim=args.embed_dim, hidden_dim=256
        ).to(device)

        predictor = Predictor(
            embed_dim=args.embed_dim, action_dim=2,
            hidden_dim=args.embed_dim, n_layers=args.pred_layers,
            n_heads=args.pred_heads, mlp_dim=args.embed_dim * 4,
            discrete_actions=False, max_seq_len=args.seq_len,
        ).to(device)

        sigreg = SIGReg(embed_dim=args.embed_dim).to(device)

        enc_p = sum(p.numel() for p in encoder.parameters())
        pred_p = sum(p.numel() for p in predictor.parameters())
        print(f"  Model: {(enc_p + pred_p) / 1e6:.1f}M params "
              f"(encoder={enc_p / 1e6:.1f}M, predictor={pred_p / 1e6:.1f}M)")

        t_start = time.time()
        history = train_lewm(
            encoder, predictor, sigreg, dataset, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            lambd=args.lambd, pixel_mode=False)
        train_time = time.time() - t_start

        print(f"\n  Evaluating rollout quality...")
        errors = evaluate_rollout(
            encoder, predictor, dataset, device,
            n_rollouts=100, horizon=10, pixel_mode=False)

        results["state"] = {
            "train_time": train_time,
            "final_pred_loss": history["pred_loss"][-1],
            "final_sigreg_loss": history["sigreg_loss"][-1],
            "avg_nan_pct": np.mean(history["nan_batches"]),
            "rollout_error_mean": np.mean(errors) if errors else None,
            "rollout_error_std": np.std(errors) if errors else None,
            "params": enc_p + pred_p,
            "avg_epoch_time": np.mean(history["epoch_time"]),
        }

        torch.save({
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "config": vars(args),
            "history": history,
        }, save_dir / "lewm_mlp_pusht.pt")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    for mode, r in results.items():
        print(f"\n  [{mode.upper()}]")
        print(f"    Params: {r['params'] / 1e6:.1f}M")
        print(f"    Train time: {r['train_time']:.1f}s")
        print(f"    Avg epoch time: {r['avg_epoch_time']:.1f}s")
        print(f"    Final pred loss: {r['final_pred_loss']:.6f}")
        print(f"    Final SIGReg loss: {r['final_sigreg_loss']:.4f}")
        print(f"    NaN batches: {r['avg_nan_pct']:.1f}%")
        if r['rollout_error_mean'] is not None:
            print(f"    Rollout error: {r['rollout_error_mean']:.4f} ± {r['rollout_error_std']:.4f}")

    # Save summary
    with open(save_dir / "gpu_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {save_dir}/gpu_training_results.json")


if __name__ == "__main__":
    main()
