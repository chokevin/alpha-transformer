# alpha-transformer

**Distributed data collection for [LeWM](https://github.com/lucas-maes/le-wm) world models using Kubernetes [agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) pods.**

The official LeWM repo trains world models from pre-collected datasets. This repo solves the step before that: **collecting large-scale trajectory datasets in parallel across AKS pods**, outputting HDF5 files that plug directly into the LeWM/[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) training pipeline.

## Why this exists

LeWM's key claim is "15M params, single GPU, few hours." The model training is already fast. The bottleneck is **data collection** — especially for pixel-based environments like Push-T where rendering takes ~0.3-0.7s per step.

| | Local | 50 AKS Pods |
|---|---|---|
| Push-T (96×96 pixels) | 1.5 eps/s | **~75 eps/s** |
| 5000 episodes | ~55 min | **~1 min** |

This repo dispatches self-contained episode scripts to isolated [agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) pods, collects trajectories as JSON, and saves HDF5 datasets compatible with `stable-worldmodel.data.HDF5Dataset`.

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Collect Push-T data locally (1.5 eps/s)
python collect_distributed.py --env pusht --episodes 500 --local --format hdf5

# Collect on AKS (50x faster for pixel environments)
./sandbox/deploy.sh
python collect_distributed.py --env pusht --episodes 5000 --batch 15 --parallel 15 --format hdf5

# Output plugs directly into stable-worldmodel
python -c "
from stable_worldmodel.data import HDF5Dataset
ds = HDF5Dataset('pusht_5000', num_steps=16, cache_dir='results')
print(f'{len(ds)} training samples ready')
"
```

## Architecture

```
┌──────────────────────────────┐
│  collect_distributed.py      │
│  (local or AKS pods)         │
│                              │     ┌───────────────┐
│  Dispatches episode scripts  │────▶│ AKS Pod 1     │──▶ trajectory
│  to sandbox pods in parallel │     │ AKS Pod 2     │──▶ trajectory
│                              │     │ ...           │
│  Collects JSON trajectories  │◀────│ AKS Pod N     │──▶ trajectory
│  Saves HDF5 (stable-wm fmt) │     └───────────────┘
└──────────┬───────────────────┘
           │ results/pusht_5000.h5
           ▼
┌──────────────────────────────┐     ┌───────────────┐
│  le-wm / stable-worldmodel   │     │ Single GPU    │
│  train.py                    │────▶│ LeWM training │
│  eval.py                     │     │ CEM planning  │
└──────────────────────────────┘     └───────────────┘
```

## Supported Environments

| Environment | Obs | Actions | Local Speed | AKS Speedup | Sandbox Deps |
|---|---|---|---|---|---|
| **Push-T** | 96×96 pixels + agent_pos | 2D continuous | 1.5 eps/s | ~50× | Custom image (gym-pusht, pymunk) |
| **Snake** | 300-dim grid | Discrete(4) | 34 eps/s | N/A (already fast) | None (pure Python) |

## HDF5 Output Format

Compatible with [`stable_worldmodel.data.HDF5Dataset`](https://github.com/galilai-group/stable-worldmodel):

```
ep_len:     (n_episodes,) int64     — length of each episode
ep_offset:  (n_episodes,) int64     — cumulative row offset
pixels:     (N, 96, 96, 3) uint8    — pixel observations (if applicable)
agent_pos:  (N, 2) float32          — agent position
action:     (N, 2) float32          — actions taken
reward:     (N,) float32            — rewards
step_idx:   (N,) int32              — timestep within episode
ep_idx:     (N,) int32              — episode index
```

## AKS Deployment

Requires an AKS cluster with [agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) controller installed.

```bash
# 1. Install agent-sandbox controller (one-time)
kubectl apply --server-side -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/v0.2.1/manifest.yaml
kubectl apply --server-side -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/v0.2.1/extensions.yaml

# 2. Deploy sandbox template + warm pool + router
./sandbox/deploy.sh

# 3. For Push-T: build custom image with deps (one-time)
az acr build --registry <your-acr> --image alpha-sandbox:latest sandbox/

# 4. Collect
python collect_distributed.py --env pusht --episodes 5000 \
    --batch 15 --parallel 15 --format hdf5

# 5. Clean up when done
./sandbox/deploy.sh --teardown
```

## Project Structure

```
alpha-transformer/
├── collect_distributed.py     # Core: distributed data collection
├── sandbox/
│   ├── Dockerfile             # Custom sandbox image (Push-T deps)
│   ├── sandbox-template.yaml  # K8s SandboxTemplate
│   ├── warm-pool.yaml         # Pre-warmed pods
│   └── deploy.sh              # One-command AKS setup
├── train_lewm_pusht.py        # LeWM training for Push-T
├── train_lewm_snake.py        # LeWM training for Snake
├── train_snake.py             # Decision Transformer baseline
├── model/
│   ├── encoder.py             # LeWM encoder
│   ├── predictor.py           # LeWM predictor (AdaLN transformer)
│   ├── sigreg.py              # SIGReg anti-collapse regularizer
│   └── transformer.py         # Decision Transformer (comparison)
├── envs/
│   ├── snake.py               # Snake Gym environment
│   └── commodities.py         # Commodity trading Gym environment
├── planner.py                 # CEM planner
└── results/                   # Saved models + HDF5 datasets
```

## Relationship to Other Repos

| Repo | Role | What it provides |
|---|---|---|
| [le-wm](https://github.com/lucas-maes/le-wm) | Model | LeWM architecture + training scripts |
| [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | Framework | HDF5Dataset, CEM/MPPI solvers, environments, eval |
| [stable-pretraining](https://github.com/galilai-group/stable-pretraining) | Framework | ViT backbone, Lightning training, monitoring callbacks |
| [sandbox-arena](https://github.com/chokevin/sandbox-arena) | Platform | Additional Gym environments, deployment scripts |
| **alpha-transformer (this repo)** | **Data pipeline** | **Distributed collection → HDF5 → feeds into all of the above** |
