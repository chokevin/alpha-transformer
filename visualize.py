#!/usr/bin/env python3
"""Visualize training results and agent performance.

Usage:
    # Visualize latest commodity training
    python visualize.py --training trajectories/commodity_training_*.json

    # Visualize a live evaluation run
    python visualize.py --eval --episodes 20
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path


def plot_training_history(history_file: str, output: str = "results/training_curve.png"):
    """Plot training curve from a training history JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_file) as f:
        data = json.load(f)

    history = data["history"]
    episodes = [h["episode"] for h in history]
    returns = [h["avg_return"] for h in history]
    sharpes = [h.get("avg_sharpe", 0) for h in history]
    drawdowns = [h.get("avg_drawdown", 0) for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Commodity Trading RL — Training Progress", fontsize=14, fontweight="bold")

    # Returns
    axes[0].plot(episodes, returns, "b-", linewidth=2, label="Avg Return %")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    if "baseline_avg_return" in data:
        axes[0].axhline(y=data["baseline_avg_return"], color="green",
                        linestyle="--", alpha=0.7, label=f"Buy-Hold ({data['baseline_avg_return']}%)")
    axes[0].fill_between(episodes, returns, 0, alpha=0.1,
                         color="blue" if returns[-1] > 0 else "red")
    axes[0].set_ylabel("Return (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sharpe
    if any(s != 0 for s in sharpes):
        axes[1].plot(episodes, sharpes, "purple", linewidth=2, label="Avg Sharpe")
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].axhline(y=1.0, color="green", linestyle=":", alpha=0.5, label="Sharpe=1 (good)")
        axes[1].set_ylabel("Sharpe Ratio")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Drawdown
    if any(d != 0 for d in drawdowns):
        axes[2].plot(episodes, drawdowns, "red", linewidth=2, label="Avg Drawdown %")
        axes[2].fill_between(episodes, drawdowns, 0, alpha=0.2, color="red")
        axes[2].set_ylabel("Drawdown (%)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    axes[2].set_xlabel("Training Episodes")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"  Saved: {output}")
    plt.close()


def plot_evaluation(episodes: int = 20, output: str = "results/evaluation.png"):
    """Run evaluation episodes and plot portfolio performance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from envs.commodities import CommodityTradingEnv

    # Load trained policy
    policy_path = "agents/trained_commodity_policy.npz"
    if not Path(policy_path).exists():
        print(f"No trained policy found at {policy_path}. Train first!")
        sys.exit(1)

    data = np.load(policy_path)
    w1, b1 = data["w1"], data["b1"]
    w2, b2 = data["w2"], data["b2"]
    w_mu, b_mu = data["w_mu"], data["b_mu"]
    log_std = data["log_std"]

    def forward(obs):
        h1 = np.tanh(obs @ w1 + b1)
        h2 = np.tanh(h1 @ w2 + b2)
        mu = np.tanh(h2 @ w_mu + b_mu)
        return mu

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Commodity Trading RL — Evaluation", fontsize=14, fontweight="bold")

    # Run episodes
    all_portfolio_curves = []
    all_returns = []
    all_positions_history = []

    for ep in range(episodes):
        env = CommodityTradingEnv(n_days=252)
        obs, _ = env.reset()
        portfolio_curve = [100000]
        positions_history = {name: [] for name in env.commodity_names}

        done = False
        while not done:
            action = forward(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            portfolio_curve.append(info["portfolio_value"])
            for name in env.commodity_names:
                positions_history[name].append(info["positions"][name])
            done = terminated or truncated

        all_portfolio_curves.append(portfolio_curve)
        all_returns.append(info.get("total_return", 0))
        all_positions_history.append(positions_history)

    # Also run buy-and-hold
    bh_curves = []
    for ep in range(episodes):
        env = CommodityTradingEnv(n_days=252)
        obs, _ = env.reset()
        bh_curve = [100000]
        done = False
        while not done:
            action = np.ones(4)
            obs, r, terminated, truncated, info = env.step(action)
            bh_curve.append(info["portfolio_value"])
            done = terminated or truncated
        bh_curves.append(bh_curve)

    # Plot 1: Portfolio curves
    ax = axes[0, 0]
    for curve in all_portfolio_curves:
        ax.plot(curve, alpha=0.2, color="blue", linewidth=0.5)
    avg_curve = np.mean([c[:min(len(c) for c in all_portfolio_curves)]
                         for c in all_portfolio_curves], axis=0)
    ax.plot(avg_curve, color="blue", linewidth=2, label="RL Agent (avg)")

    for curve in bh_curves:
        ax.plot(curve, alpha=0.1, color="green", linewidth=0.5)
    avg_bh = np.mean([c[:min(len(c) for c in bh_curves)]
                      for c in bh_curves], axis=0)
    ax.plot(avg_bh, color="green", linewidth=2, label="Buy & Hold (avg)")

    ax.axhline(y=100000, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Return distribution
    ax = axes[0, 1]
    ax.hist(all_returns, bins=20, alpha=0.7, color="blue", label="RL Agent")
    bh_returns = [(c[-1] / c[0] - 1) * 100 for c in bh_curves]
    ax.hist(bh_returns, bins=20, alpha=0.5, color="green", label="Buy & Hold")
    ax.axvline(x=0, color="gray", linestyle="--")
    ax.axvline(x=np.mean(all_returns), color="blue", linestyle="-",
               label=f"Agent avg: {np.mean(all_returns):.1f}%")
    ax.axvline(x=np.mean(bh_returns), color="green", linestyle="-",
               label=f"B&H avg: {np.mean(bh_returns):.1f}%")
    ax.set_title("Return Distribution")
    ax.set_xlabel("Total Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Average position sizes over time
    ax = axes[1, 0]
    colors = {"gold": "#FFD700", "oil": "#333333", "wheat": "#DEB887", "natgas": "#87CEEB"}
    for name in ["gold", "oil", "wheat", "natgas"]:
        all_pos = []
        for ph in all_positions_history:
            if name in ph and ph[name]:
                all_pos.append(ph[name])
        if all_pos:
            min_len = min(len(p) for p in all_pos)
            avg_pos = np.mean([p[:min_len] for p in all_pos], axis=0)
            ax.plot(avg_pos, label=name.title(), color=colors[name], linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Average Position Sizes")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Units Held")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary stats
    ax = axes[1, 1]
    ax.axis("off")
    stats = [
        f"RL Agent ({episodes} episodes)",
        f"─────────────────────────",
        f"Avg Return:    {np.mean(all_returns):>7.2f}%",
        f"Std Return:    {np.std(all_returns):>7.2f}%",
        f"Win Rate:      {sum(1 for r in all_returns if r > 0)/len(all_returns)*100:>6.0f}%",
        f"Best:          {np.max(all_returns):>7.2f}%",
        f"Worst:         {np.min(all_returns):>7.2f}%",
        f"",
        f"Buy & Hold",
        f"─────────────────────────",
        f"Avg Return:    {np.mean(bh_returns):>7.2f}%",
        f"Std Return:    {np.std(bh_returns):>7.2f}%",
        f"",
        f"Alpha:         {np.mean(all_returns) - np.mean(bh_returns):>+7.2f}%",
    ]
    ax.text(0.1, 0.95, "\n".join(stats), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"  Saved: {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize training and evaluation")
    parser.add_argument("--training", help="Path to training history JSON")
    parser.add_argument("--eval", action="store_true", help="Run evaluation and plot")
    parser.add_argument("--episodes", type=int, default=20, help="Eval episodes")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    if args.training:
        plot_training_history(args.training,
                             output=f"{args.output_dir}/training_curve.png")

    if args.eval:
        plot_evaluation(episodes=args.episodes,
                        output=f"{args.output_dir}/evaluation.png")

    if not args.training and not args.eval:
        # Auto-find latest training file
        training_files = sorted(Path("trajectories").glob("commodity_training_*.json"))
        if training_files:
            latest = str(training_files[-1])
            print(f"Using latest training: {latest}")
            plot_training_history(latest, output=f"{args.output_dir}/training_curve.png")
            plot_evaluation(episodes=20, output=f"{args.output_dir}/evaluation.png")
        else:
            print("No training data found. Run train_commodities.py first.")


if __name__ == "__main__":
    main()
