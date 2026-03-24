# alpha-transformer Agent Instructions

## Training

Always use the full pipeline for initial training:
```bash
python train.py --phase all
```

For iteration on RL only (after SFT):
```bash
python train.py --phase rl --episodes 5000
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

For large-scale RL, use sandbox-arena's distributed runner
which dispatches episodes to AKS sandbox pods:
```bash
cd ../sandbox-arena
python train_distributed.py --episodes 10000 --batch 50 --parallel 50
```
