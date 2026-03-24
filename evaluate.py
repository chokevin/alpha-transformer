#!/usr/bin/env python3
"""Evaluate trained model against baselines."""

import json
import numpy as np
import torch
from pathlib import Path
from envs.commodities import CommodityTradingEnv
from envs.features import STATE_DIM
from model.transformer import TradingTransformer
from model.policy import run_episode, load_checkpoint


def load_model(path="results/trading_transformer.pt"):
    model, metadata = load_checkpoint(path, TradingTransformer)
    model.eval()
    return model


def evaluate(model, n_episodes=100, target_return=20.0):
    results = []
    for _ in range(n_episodes):
        env = CommodityTradingEnv(n_days=252)
        info, _ = run_episode(model, env, target_return=target_return,
                              deterministic=True)
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
    print(f"Model: {model.param_count():,} parameters")
    print("Evaluating (100 episodes each)...\n")

    agent_results = evaluate(model, 100)
    bh_results = run_baseline(100)

    agent_returns = [r.get("total_return", 0) for r in agent_results]
    bh_returns = [r.get("total_return", 0) for r in bh_results]
    agent_sharpes = [r.get("sharpe", 0) for r in agent_results]
    agent_drawdowns = [r.get("max_drawdown", 0) for r in agent_results]

    print(f"{'='*60}")
    print(f"{'Model':<20} {'Return':>8} {'Std':>8} {'Win%':>6} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"{'='*60}")
    print(f"{'Transformer':<20} {np.mean(agent_returns):>+7.2f}% {np.std(agent_returns):>7.2f}% "
          f"{sum(1 for r in agent_returns if r > 0)/len(agent_returns)*100:>5.0f}% "
          f"{np.mean(agent_sharpes):>+7.3f} "
          f"{np.mean(agent_drawdowns):>7.2f}%")
    print(f"{'Buy & Hold':<20} {np.mean(bh_returns):>+7.2f}% {np.std(bh_returns):>7.2f}% "
          f"{'—':>6} {'—':>8} {'—':>8}")
    print(f"{'='*60}")
    print(f"{'Alpha':<20} {np.mean(agent_returns) - np.mean(bh_returns):>+7.2f}%")

    # Save results
    results_path = Path("results/evaluation.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "agent": {"returns": agent_returns, "sharpes": agent_sharpes,
                      "drawdowns": agent_drawdowns},
            "baseline": {"returns": bh_returns},
        }, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
