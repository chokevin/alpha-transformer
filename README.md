# alpha-transformer

Decision Transformer for multi-commodity trading. Trains a causal transformer to trade Gold, Oil, Wheat, and Natural Gas using the pretrain → SFT → RL pipeline.

Uses [sandbox-arena](https://github.com/chokevin/sandbox-arena) as the RL environment.

## Architecture

```
┌────────────────────────────────┐      ┌──────────────────────────────┐
│  alpha-transformer (this repo) │      │  sandbox-arena (environment) │
│                                │      │                              │
│  Decision Transformer          │ ───> │  Commodity trading env       │
│  168K params, 3 layers         │ acts │  Gold, Oil, Wheat, NatGas    │
│                                │      │  Realistic price dynamics    │
│  Phase 1: Collect trajectories │ <─── │  Technical indicators        │
│  Phase 2: SFT on good trajs   │ rwds │  Transaction costs           │
│  Phase 3: RL fine-tuning       │      │                              │
└────────────────────────────────┘      └──────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Full pipeline: collect → SFT → RL → evaluate
python train.py --phase all

# Just RL fine-tuning (after SFT)
python train.py --phase rl --episodes 2000

# Evaluate against baselines
python evaluate.py

# Visualize results
python visualize.py
```

## Training Pipeline

### Phase 1: Collect Trajectories
Gather episodes from heuristic policies. Filter for profitable ones to avoid training on bad data.

### Phase 2: Supervised Fine-Tuning (SFT)
Train the transformer to predict actions from (state, return-to-go) sequences. The model learns to imitate profitable trading patterns.

### Phase 3: RL Fine-Tuning
REINFORCE against the live commodity environment. The model learns to improve beyond the SFT policy.

## Model

Causal transformer that treats trading as sequence modeling:

| Component | Details |
|-----------|---------|
| Architecture | Causal transformer encoder |
| Parameters | 168K |
| Layers | 3 |
| Heads | 4 |
| Hidden dim | 64 |
| Sequence length | 60 timesteps |
| Input | Interleaved (return-to-go, state, action) tokens |
| Output | Position sizes for 4 commodities (continuous [-1, 1]) |

## Results

| Model | Avg Return | Std | Win Rate | Sharpe | Alpha vs B&H |
|-------|-----------|-----|----------|--------|--------------|
| Buy & Hold | +3.1% | 7.2% | — | — | baseline |
| Simple NN (REINFORCE) | -0.01% | 7.5% | 56% | 0.002 | -3.1% |
| Decision Transformer (v1) | -0.17% | 0.3% | 32% | -0.49 | -3.2% |
| **Target** | **>5%** | **<10%** | **>55%** | **>0.5** | **positive** |

## What needs to improve

1. **Better collection data** — filter for top 20% trajectories, add profitable heuristics
2. **More RL episodes** — scale to 5000+ with sandbox-arena parallelism
3. **Reward shaping** — penalize concentration, reward Sharpe not just returns
4. **Features** — add volume, volatility regime, cross-commodity correlations
5. **Architecture** — try MoE, longer context, attention to specific commodities

## Project Structure

```
alpha-transformer/
├── model/
│   ├── transformer.py      # Decision Transformer architecture
│   └── policy.py           # Action selection + exploration
├── envs/
│   ├── commodities.py      # Commodity trading Gym env (from sandbox-arena)
│   └── features.py         # Technical feature engineering
├── configs/
│   └── default.yaml        # Hyperparameters
├── train.py                # 3-phase training pipeline
├── evaluate.py             # Evaluation against baselines
├── visualize.py            # Training curves + portfolio charts
├── requirements.txt
└── results/                # Saved models + training logs
```

## Relationship to sandbox-arena

This repo is the **model**. [sandbox-arena](https://github.com/chokevin/sandbox-arena) is the **environment/platform**.

- sandbox-arena provides the `CommodityTradingEnv` Gym environment
- sandbox-arena handles distributed episode collection across AKS pods
- This repo handles model architecture, training, and evaluation
- To train distributed: use `sandbox-arena/train_distributed.py` with this model's weights
