"""Policy utilities for the Decision Transformer.

Handles:
  - Building sequence context for inference (rolling window)
  - Action sampling with exploration
  - Running full episodes (inference loop)
  - Saving/loading checkpoints
"""

import numpy as np
import torch


def build_context(states_buf, actions_buf, rewards_buf, model, target_return=20.0):
    """Build input tensors for the transformer from episode buffers.

    Maintains a rolling window of max_seq_len steps and computes
    return-to-go from the target return minus accumulated rewards.

    Args:
        states_buf: list of observation arrays
        actions_buf: list of action arrays
        rewards_buf: list of scalar rewards
        model: TradingTransformer (for max_seq_len, action_dim)
        target_return: desired episode return (%)

    Returns:
        (states, actions, rtgs) tensors, each (1, seq_len, dim)
    """
    seq_len = min(len(states_buf), model.max_seq_len)

    s = torch.tensor(
        np.array(states_buf[-seq_len:]), dtype=torch.float32
    ).unsqueeze(0)

    if actions_buf:
        a_len = min(len(actions_buf), seq_len)
        a = torch.tensor(
            np.array(actions_buf[-a_len:]), dtype=torch.float32
        ).unsqueeze(0)
        if a.shape[1] < s.shape[1]:
            pad = torch.zeros(1, s.shape[1] - a.shape[1], model.action_dim)
            a = torch.cat([pad, a], dim=1)
    else:
        a = torch.zeros(1, seq_len, model.action_dim)

    current_rtg = (target_return - sum(rewards_buf)) / 100.0
    r = torch.full((1, seq_len, 1), current_rtg, dtype=torch.float32)

    return s, a, r


def run_episode(model, env, target_return=20.0, deterministic=True):
    """Run a single episode with the model.

    Returns:
        info: dict with episode results (total_return, sharpe, etc.)
        trajectory: dict with states, actions, rewards lists
    """
    obs, _ = env.reset()
    states_buf, actions_buf, rewards_buf = [], [], []
    done = False

    while not done:
        states_buf.append(obs)
        s, a, r = build_context(states_buf, actions_buf, rewards_buf,
                                model, target_return)

        with torch.no_grad():
            action = model.get_action(s, a, r, deterministic=deterministic)
        action = action.squeeze(0).numpy()

        actions_buf.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_buf.append(reward)
        done = terminated or truncated

    trajectory = {
        "states": [s.tolist() for s in states_buf],
        "actions": [a.tolist() for a in actions_buf],
        "rewards": rewards_buf,
        "total_return": info.get("total_return", 0),
    }

    return info, trajectory


def save_checkpoint(model, path, metadata=None):
    """Save model checkpoint with optional metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "hidden_dim": model.hidden_dim,
            "max_seq_len": model.max_seq_len,
        },
    }
    if metadata:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, path)


def load_checkpoint(path, model_class=None):
    """Load model from checkpoint.

    Args:
        path: path to .pt file
        model_class: TradingTransformer class (imported by caller)

    Returns:
        model, metadata
    """
    checkpoint = torch.load(path, weights_only=False)

    if model_class and "config" in checkpoint:
        config = checkpoint["config"]
        model = model_class(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif model_class:
        model = model_class()
        model.load_state_dict(checkpoint)
    else:
        return checkpoint, checkpoint.get("metadata")

    model.eval()
    return model, checkpoint.get("metadata")
