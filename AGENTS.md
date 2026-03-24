# alpha-transformer Agent Instructions

## Training

Always use the full pipeline for initial training:
```bash
source .venv/bin/activate
python train.py --phase all
```

For iteration on RL only (after SFT):
```bash
python train.py --phase rl --rl-episodes 5000
```

## Environment

The commodity trading environment comes from sandbox-arena.
Clone it alongside this repo:
```
~/dev/
├── alpha-transformer/    # this repo (model)
└── sandbox-arena/        # environment + platform
```

## Distributed Training

For large-scale RL, use AKS sandbox pods for parallel episode collection:
```bash
python train_distributed.py --episodes 10000 --batch 50 --parallel 50
```

Prerequisites:
1. AKS cluster with agent-sandbox controller
2. `kubectl apply -f ../sandbox-arena/sandbox/sandbox-template.yaml`
3. `kubectl apply -f ../sandbox-arena/sandbox/warm-pool.yaml`
4. `pip install k8s-agent-sandbox`

## Config

Hyperparameters live in `configs/default.yaml`. CLI args override config values.

## Code Structure

- `model/transformer.py` — TradingTransformer architecture (899K params)
- `model/policy.py` — Inference helpers, checkpoint save/load
- `envs/features.py` — 52-dim feature engineering (per-commodity + cross-commodity + portfolio)
- `envs/commodities.py` — CommodityTradingEnv with Sharpe-based reward shaping
- `train.py` — 3-phase local training (collect → SFT → RL)
- `train_distributed.py` — Distributed RL via AKS sandbox pods
