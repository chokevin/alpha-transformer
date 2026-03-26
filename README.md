# alpha-transformer

**LeWM world model training with a novel SIGReg hyperparameter finding: λ=0.01–0.032 beats the paper's default λ=0.1 by 29% on held-out test data.**

Built on [LeWM](https://github.com/lucas-maes/le-wm) (Le World Model) with distributed data collection via Kubernetes [agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) pods, GPU training on AKS, and HDF5 output compatible with [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).

## Key Finding: λ Ablation

The LeWM paper uses λ=0.1 for the SIGReg regularizer weight. Our controlled ablation on Push-T (ViT-192, 6.6M params, 200 episodes, 80/20 train/test split, T4 GPU) shows lower λ values produce significantly better world models:

| λ | Test H=1 | Test H=5 | Test H=10 | vs Paper |
|---|---|---|---|---|
| **0.010** | **0.0015** | **0.0014** | **0.0015** | **-29%** ↓ |
| **0.032** | **0.0015** | **0.0015** | **0.0015** | **-29%** ↓ |
| 0.100 (paper) | 0.0020 | 0.0020 | 0.0021 | baseline |
| 0.320 | 0.0042 | 0.0042 | 0.0043 | +105% ↑ |

- Evaluated on **held-out test episodes** (not training data)
- All models trained identically (same data, architecture, optimizer, 50 epochs)
- 0% NaN batches on GPU (vs 17% on CPU)
- Train/test errors are nearly identical → model generalizes, does not memorize

**Why this matters:** SIGReg (Sketched Isotropic Gaussian Regularizer) prevents representation collapse in JEPA architectures. Too much regularization (λ=0.32) degrades prediction quality by over-constraining the latent space. The paper's default λ=0.1 is 29% worse than optimal.

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# GPU training (single script)
python train_lewm_gpu.py --mode pixel --episodes 200 --epochs 50 --lambd 0.032

# λ ablation experiment
python experiment_lambda.py --episodes 200 --lambdas 0.01,0.032,0.1,0.32

# Distributed data collection (50x faster on AKS)
python collect_distributed.py --env pusht --episodes 5000 --batch 15 --parallel 15
```

## GPU Training on AKS

```bash
# 1. Build training image
az acr build --registry <your-acr> --image alpha-train:latest -f sandbox/Dockerfile.gpu .

# 2. Submit GPU training job (T4/V100/A100)
kubectl apply -f sandbox/gpu-train-job.yaml

# 3. Run λ ablation
kubectl apply -f sandbox/lambda-ablation-job.yaml
```

## Results

### GPU Training (T4)

| Metric | CPU | GPU (T4) |
|---|---|---|
| Pred loss (final) | 0.0006 | **0.0000004** |
| Loss reduction | 250× | **59,000×** |
| NaN batches | 17% | **0%** |
| Epoch time | ~45s | **21.4s** |

### Distributed Data Collection

| | Local | 50 AKS Pods |
|---|---|---|
| Push-T (96×96 pixels) | 1.5 eps/s | **~75 eps/s** |
| 5000 episodes | ~55 min | **~1 min** |

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
│  train_lewm_gpu.py           │     │ T4/V100/A100  │
│  experiment_lambda.py        │────▶│ ViT encoder   │
│  (LeWM: encoder+predictor)   │     │ + predictor   │
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
├── experiment_lambda.py        # λ ablation: the key finding (λ=0.032 > 0.1)
├── train_lewm_gpu.py           # GPU training: ViT pixel world model
├── collect_distributed.py      # Distributed data collection (local + AKS)
├── sandbox/
│   ├── Dockerfile.gpu          # PyTorch CUDA training image
│   ├── gpu-train-job.yaml      # K8s Job for GPU training
│   ├── lambda-ablation-job.yaml# K8s Job for λ experiment
│   ├── Dockerfile              # Custom sandbox image (Push-T deps)
│   ├── sandbox-template.yaml   # K8s SandboxTemplate
│   ├── warm-pool.yaml          # Pre-warmed pods
│   └── deploy.sh               # One-command AKS setup
├── model/
│   ├── encoder.py              # MLP + ViT encoders
│   ├── predictor.py            # Transformer predictor (AdaLN)
│   └── sigreg.py               # SIGReg anti-collapse regularizer
├── planner.py                  # CEM planner
├── train_lewm_pusht.py         # Push-T training (state-based)
├── sweep_lambda.py             # Parallel λ sweep on AKS
└── results/                    # Saved models + datasets
```

## Relationship to Other Repos

| Repo | Role | What it provides |
|---|---|---|
| [le-wm](https://github.com/lucas-maes/le-wm) | Model | LeWM architecture + training scripts |
| [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | Framework | HDF5Dataset, CEM/MPPI solvers, environments, eval |
| [stable-pretraining](https://github.com/galilai-group/stable-pretraining) | Framework | ViT backbone, Lightning training, monitoring callbacks |
| [sandbox-arena](https://github.com/chokevin/sandbox-arena) | Platform | Additional Gym environments, deployment scripts |
| **alpha-transformer (this repo)** | **Data pipeline** | **Distributed collection → HDF5 → feeds into all of the above** |
