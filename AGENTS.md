# alpha-transformer Agent Instructions

## What This Repo Does
Distributed data collection for LeWM world models. Collects trajectories
from Gym environments via AKS agent-sandbox pods, outputs HDF5 datasets
compatible with stable-worldmodel.

## Data Collection

```bash
# Local collection
source .venv/bin/activate
python collect_distributed.py --env pusht --episodes 500 --local --format hdf5

# Distributed via AKS (deploy infra first: ./sandbox/deploy.sh)
python collect_distributed.py --env pusht --episodes 5000 --batch 15 --parallel 15 --format hdf5
```

## Training (LeWM)

Uses stable-worldmodel for data loading:
```python
from stable_worldmodel.data import HDF5Dataset
ds = HDF5Dataset('pusht_5000', num_steps=16, cache_dir='results')
```

Or our standalone scripts:
```bash
python train_lewm_pusht.py --phase all --episodes 500
python train_lewm_snake.py --phase all --episodes 3000
```

## AKS Deployment

```bash
# Prerequisites: AKS cluster + agent-sandbox controller
./sandbox/deploy.sh                     # Apply K8s configs
python collect_distributed.py ...       # Collect data
./sandbox/deploy.sh --teardown          # Clean up
```

## Adding a New Environment

1. Write a self-contained episode script (see `SNAKE_EPISODE_SCRIPT` in collect_distributed.py)
2. Add it to `collect_distributed.py` (local + distributed paths)
3. If it needs native deps, add them to `sandbox/Dockerfile` and rebuild the image
4. Test locally first: `--local`, then on AKS
