#!/usr/bin/env python3
"""Visualize training results and agent performance.

Usage:
    # Visualize latest RL training history
    python visualize.py --training results/rl_history_*.json

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

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Commodity Trading — Decision Transformer Training", fontsize=14, fontweight="bold")

    # Returns
    axes[0].plot(episodes, returns, "b-", linewidth=2, label="Avg Return %")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].fill_between(episodes, returns, 0, alpha=0.1,
                         color="blue" if returns[-1] > 0 else "red")
    axes[0].set_ylabel("Return (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sharpe
    if any(s != 0 for s in sharpes):
        axes[1].plot(episodes, sharpes, "purple", linewidth=2, label="Avg Sharpe")
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].axhline(y=0.5, color="green", linestyle=":", alpha=0.5, label="Sharpe=0.5 (target)")
        axes[1].set_ylabel("Sharpe Ratio")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Training Episodes")

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
    from model.transformer import TradingTransformer
    from model.policy import run_episode, load_checkpoint

    model_path = "results/trading_transformer.pt"
    if not Path(model_path).exists():
        print(f"No trained model found at {model_path}. Train first!")
        sys.exit(1)

    model, _ = load_checkpoint(model_path, TradingTransformer)
    model.eval()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Commodity Trading — Decision Transformer Evaluation", fontsize=14, fontweight="bold")

    # Run episodes
    all_returns = []
    all_sharpes = []

    for ep in range(episodes):
        env = CommodityTradingEnv(n_days=252)
        info, _ = run_episode(model, env, deterministic=True)
        all_returns.append(info.get("total_return", 0))
        all_sharpes.append(info.get("sharpe", 0))

    # Buy-and-hold baseline
    bh_returns = []
    for ep in range(episodes):
        env = CommodityTradingEnv(n_days=252)
        obs, _ = env.reset()
        done = False
        while not done:
            obs, r, terminated, truncated, info = env.step(np.ones(4))
            done = terminated or truncated
        bh_returns.append(info.get("total_return", 0))

    # Plot 1: Return distribution
    ax = axes[0, 0]
    ax.hist(all_returns, bins=20, alpha=0.7, color="blue", label="Transformer")
    ax.hist(bh_returns, bins=20, alpha=0.5, color="green", label="Buy & Hold")
    ax.axvline(x=0, color="gray", linestyle="--")
    ax.axvline(x=np.mean(all_returns), color="blue", linestyle="-",
               label=f"Agent avg: {np.mean(all_returns):+.1f}%")
    ax.axvline(x=np.mean(bh_returns), color="green", linestyle="-",
               label=f"B&H avg: {np.mean(bh_returns):+.1f}%")
    ax.set_title("Return Distribution")
    ax.set_xlabel("Total Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Sharpe distribution
    ax = axes[0, 1]
    ax.hist(all_sharpes, bins=20, alpha=0.7, color="purple", label="Sharpe Ratio")
    ax.axvline(x=0, color="gray", linestyle="--")
    ax.axvline(x=np.mean(all_sharpes), color="purple", linestyle="-",
               label=f"Avg: {np.mean(all_sharpes):.3f}")
    ax.axvline(x=0.5, color="green", linestyle=":", label="Target (0.5)")
    ax.set_title("Sharpe Ratio Distribution")
    ax.set_xlabel("Sharpe Ratio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Returns scatter (agent vs baseline)
    ax = axes[1, 0]
    ax.scatter(bh_returns[:len(all_returns)], all_returns, alpha=0.5, color="blue")
    lim = max(abs(min(all_returns + bh_returns)), abs(max(all_returns + bh_returns))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Buy & Hold Return (%)")
    ax.set_ylabel("Transformer Return (%)")
    ax.set_title("Agent vs Baseline (per episode)")
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary stats
    ax = axes[1, 1]
    ax.axis("off")
    alpha = np.mean(all_returns) - np.mean(bh_returns)
    stats = [
        f"Decision Transformer ({episodes} episodes)",
        f"─────────────────────────────",
        f"Avg Return:    {np.mean(all_returns):>+7.2f}%",
        f"Std Return:    {np.std(all_returns):>7.2f}%",
        f"Win Rate:      {sum(1 for r in all_returns if r > 0)/len(all_returns)*100:>6.0f}%",
        f"Avg Sharpe:    {np.mean(all_sharpes):>+7.3f}",
        f"Best:          {np.max(all_returns):>+7.2f}%",
        f"Worst:         {np.min(all_returns):>+7.2f}%",
        f"",
        f"Buy & Hold",
        f"─────────────────────────────",
        f"Avg Return:    {np.mean(bh_returns):>+7.2f}%",
        f"Std Return:    {np.std(bh_returns):>7.2f}%",
        f"",
        f"Alpha:         {alpha:>+7.2f}%",
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
        training_files = sorted(Path("results").glob("rl_history_*.json"))
        if training_files:
            latest = str(training_files[-1])
            print(f"Using latest training: {latest}")
            plot_training_history(latest, output=f"{args.output_dir}/training_curve.png")
            plot_evaluation(episodes=20, output=f"{args.output_dir}/evaluation.png")
        else:
            print("No training data found. Run train.py --phase all first.")


if __name__ == "__main__":
    main()
