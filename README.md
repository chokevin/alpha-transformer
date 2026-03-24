# alpha-transformer

Decision Transformer for multi-commodity trading. Trains a causal transformer to trade Gold, Oil, Wheat, and Natural Gas using the pretrain → SFT → RL pipeline.

Uses [sandbox-arena](https://github.com/chokevin/sandbox-arena) as the RL environment and supports distributed training on AKS.

## Architecture

```
┌────────────────────────────────┐      ┌──────────────────────────────┐
│  alpha-transformer (this repo) │      │  sandbox-arena (environment) │
│                                │      │                              │
│  Decision Transformer          │ ───> │  Commodity trading env       │
│  899K params, 4 layers         │ acts │  Gold, Oil, Wheat, NatGas    │
│  128 hidden, 52-dim state      │      │  Realistic price dynamics    │
│                                │      │  Technical indicators        │
│  Phase 1: Collect trajectories │ <─── │  Cross-commodity features    │
│  Phase 2: SFT on top 20%      │ rwds │  Transaction costs           │
│  Phase 3: RL fine-tuning       │      │                              │
│  (PPO-style + entropy bonus)   │      │  Distributed via AKS pods   │
└────────────────────────────────┘      └──────────────────────────────┘
```

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline: collect → SFT → RL → evaluate
python train.py --phase all

# Just RL fine-tuning (after SFT)
python train.py --phase rl --rl-episodes 5000

# Distributed RL on AKS (50 parallel sandbox pods)
python train_distributed.py --episodes 5000 --batch 50 --parallel 50

# Evaluate against baselines
python evaluate.py

# Visualize results
python visualize.py
```

## Training Pipeline

### Phase 1: Collect Trajectories
Gather episodes from diverse heuristic policies (Bollinger reversion, momentum, risk-parity). Filter for top 20% by return to train only on profitable patterns.

### Phase 2: Supervised Fine-Tuning (SFT)
Train the transformer to predict actions from (state, return-to-go) sequences. Uses warmup + cosine LR decay. The model learns to imitate profitable trading patterns.

### Phase 3: RL Fine-Tuning
PPO-style clipped policy gradient against the live commodity environment with entropy bonus for exploration. Saves best model by Sharpe ratio.

## Model

Causal transformer that treats trading as sequence modeling:

| Component | Details |
|-----------|---------|
| Architecture | Causal transformer encoder (pre-norm) |
| Parameters | 899K |
| Layers | 4 |
| Heads | 4 |
| Hidden dim | 128 |
| Sequence length | 60 timesteps |
| Input | Interleaved (return-to-go, state, action) tokens |
| Output | Position sizes for 4 commodities (continuous [-1, 1]) |

## Features (52-dimensional state)

| Feature Group | Count | Details |
|---------------|-------|---------|
| Per-commodity | 10 × 4 = 40 | Returns (1/5/20d), MA ratios, realized vol, vol regime, RSI, Bollinger, momentum divergence |
| Cross-commodity | 6 | Pairwise rolling correlations |
| Portfolio state | 6 | Cash %, return, drawdown, time remaining, concentration (HHI), turnover |

## Reward Shaping

- **Sharpe contribution**: daily return / rolling volatility
- **Drawdown penalty**: penalize distance from peak
- **Concentration penalty**: Herfindahl index of position weights
- **Turnover penalty**: transaction cost awareness
- **Terminal bonus**: episode Sharpe ratio − max drawdown

## Config

All hyperparameters are in `configs/default.yaml` and loaded automatically. CLI args override config values:

```bash
python train.py --hidden 256 --layers 6  # override model size
```

## Project Structure

```
alpha-transformer/
├── model/
│   ├── transformer.py      # Decision Transformer architecture
│   └── policy.py           # Action selection, inference, checkpointing
├── envs/
│   ├── commodities.py      # Commodity trading Gym env
│   └── features.py         # Technical + cross-commodity feature engineering
├── configs/
│   └── default.yaml        # Hyperparameters
├── train.py                # 3-phase training pipeline (local)
├── train_distributed.py    # Distributed RL via AKS sandbox pods
├── evaluate.py             # Evaluation against baselines
├── visualize.py            # Training curves + portfolio charts
├── requirements.txt
└── results/                # Saved models + training logs
```

## Distributed Training on AKS

For large-scale RL, use sandbox-arena's AKS infrastructure:

```bash
# Prerequisites: AKS cluster with agent-sandbox controller
cd ../sandbox-arena
kubectl apply -f sandbox/sandbox-template.yaml
kubectl apply -f sandbox/warm-pool.yaml

# Run distributed training (50 parallel pods, ~10x faster)
cd ../alpha-transformer
python train_distributed.py --episodes 5000 --batch 50 --parallel 50
```

Each episode runs in an isolated sandbox pod. The trainer dispatches batches via `k8s-agent-sandbox`, collects trajectories, and updates the model locally.

## Results

| Model | Avg Return | Std | Win Rate | Sharpe | Alpha vs B&H |
|-------|-----------|-----|----------|--------|--------------|
| Buy & Hold | +1.9% | 9.3% | — | — | baseline |
| **Decision Transformer (SFT)** | **+0.8%** | **2.7%** | **60%** | **+0.30** | **-1.1%** |
| Target | >5% | <10% | >55% | >0.5 | positive |

**Key insight**: SFT-only outperforms SFT+RL. Policy gradient RL consistently collapses actions toward zero in this environment. See improvement roadmap below.

## Improvement Roadmap

1. **More data** — Collect 5K+ trajectories via AKS parallelism, filter top 20%
2. **DAgger** — Run model, identify failure states, collect expert demos, retrain SFT
3. **Offline RL** — Conservative Q-Learning (CQL) avoids the action collapse problem
4. **Smaller model** — Try 200K params with 5x more data (reduce overfitting)
5. **Real data** — Replace synthetic prices with historical commodity data

## Relationship to sandbox-arena

This repo is the **model**. [sandbox-arena](https://github.com/chokevin/sandbox-arena) is the **environment/platform**.

- sandbox-arena provides the `CommodityTradingEnv` Gym environment
- sandbox-arena handles distributed episode collection across AKS pods
- This repo handles model architecture, training, and evaluation
- Clone both side-by-side: `~/dev/alpha-transformer/` and `~/dev/sandbox-arena/`
