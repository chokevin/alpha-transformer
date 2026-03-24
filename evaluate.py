#!/usr/bin/env python3
"""Evaluate trained model against baselines."""

import json
import numpy as np
import torch
from pathlib import Path
from envs.commodities import CommodityTradingEnv


def load_model(path="results/trading_transformer.pt"):
    from train import TradingTransformer
    model = TradingTransformer()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def run_episode(model, env, deterministic=True):
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
                pad = torch.zeros(1, s.shape[1] - a.shape[1], model.action_dim)
                a = torch.cat([pad, a], dim=1)
        else:
            a = torch.zeros(1, seq_len, model.action_dim)
        r = torch.full((1, seq_len, 1), 0.2, dtype=torch.float32)

        with torch.no_grad():
            action = model.get_action(s, a, r, deterministic=deterministic)
        action = action.squeeze(0).numpy()
        actions_buf.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def evaluate(model, n_episodes=100):
    results = []
    for _ in range(n_episodes):
        env = CommodityTradingEnv(n_days=252)
        info = run_episode(model, env)
        results.append(info)
    return results


def run_baseline(n_episodes=100):
    """Buy and hold baseline."""
    results = []
    for _ in range(n_episodes):
        env = CommodityTradingEnv(n_days=252)
        obs, _ = env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(np.ones(4))
            done = terminated or truncated
        results.append(info)
    return results


def main():
    model_path = "results/trading_transformer.pt"
    if not Path(model_path).exists():
        print("No trained model found. Run: python train.py --phase all")
        return

    model = load_model(model_path)
    print("Evaluating (100 episodes each)...\n")

    agent_results = evaluate(model, 100)
    bh_results = run_baseline(100)

    agent_returns = [r.get("total_return", 0) for r in agent_results]
    bh_returns = [r.get("total_return", 0) for r in bh_results]
    agent_sharpes = [r.get("sharpe", 0) for r in agent_results]

    print(f"{'='*50}")
    print(f"{'Model':<20} {'Return':>8} {'Std':>8} {'Win%':>6} {'Sharpe':>8}")
    print(f"{'='*50}")
    print(f"{'Transformer':<20} {np.mean(agent_returns):>7.2f}% {np.std(agent_returns):>7.2f}% "
          f"{sum(1 for r in agent_returns if r > 0)/len(agent_returns)*100:>5.0f}% "
          f"{np.mean(agent_sharpes):>7.3f}")
    print(f"{'Buy & Hold':<20} {np.mean(bh_returns):>7.2f}% {np.std(bh_returns):>7.2f}% "
          f"{'—':>6} {'—':>8}")
    print(f"{'='*50}")
    print(f"{'Alpha':<20} {np.mean(agent_returns) - np.mean(bh_returns):>+7.2f}%")


if __name__ == "__main__":
    main()
