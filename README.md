# alpha-transformer

**Reimplementation of [LeWM](https://github.com/lucas-maes/le-wm) (Le World Model) with a novel hyperparameter finding: the paper's default SIGReg weight λ=0.1 is 29% worse than optimal (λ=0.01–0.032) on held-out test data.**

Includes GPU training on AKS (T4/V100/A100), distributed data collection via [agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) pods, and HDF5 output compatible with [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).

## Key Finding: SIGReg λ Ablation

SIGReg prevents representation collapse in JEPA world models. The [LeWM paper](https://arxiv.org/abs/2603.19312) uses λ=0.1. Our controlled ablation shows this over-regularizes the latent space:

| λ | Held-out H=1 | Held-out H=5 | Held-out H=10 | vs Paper |
|---|---|---|---|---|
| **0.010** | **0.0015** | **0.0014** | **0.0015** | **-29%** ↓ |
| **0.032** | **0.0015** | **0.0015** | **0.0015** | **-29%** ↓ |
| 0.100 (paper) | 0.0020 | 0.0020 | 0.0021 | baseline |
| 0.320 | 0.0042 | 0.0042 | 0.0043 | +105% ↑ |

**Experiment details:**
- Push-T pixel environment (96×96), ViT-192 encoder, 6.6M params
- 200 episodes, 80/20 train/test split, 50 epochs per λ
- Tesla T4 GPU on AKS, 0% NaN batches (vs 17% on CPU float32)
- Train/test errors match → model generalizes, not memorizing

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train locally (CPU or single GPU)
python train_lewm_gpu.py --mode pixel --episodes 200 --epochs 50 --lambd 0.032

# Run λ ablation
python experiment_lambda.py --episodes 200 --lambdas 0.01,0.032,0.1,0.32
```

## GPU Training on AKS

```bash
# Build training image (uses ACR Build — no local Docker needed)
az acr build --registry <your-acr> --image alpha-train:latest -f sandbox/Dockerfile.gpu .

# Submit training job to T4/V100/A100 nodepool
kubectl apply -f sandbox/gpu-train-job.yaml

# Or run the full λ ablation (~2 hours on T4)
kubectl apply -f sandbox/lambda-ablation-job.yaml
```

| Metric | CPU | GPU (T4) |
|---|---|---|
| Final pred loss | 0.0006 | **4e-7** |
| NaN batches | 17% | **0%** |
| Epoch time | ~45s | **21s** |

## Distributed Data Collection

For environments where rendering is the bottleneck (e.g., Push-T pixel), data collection can be parallelized across AKS [agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) pods:

```bash
# Local (1.5 eps/s)
python collect_distributed.py --env pusht --episodes 500 --local

# AKS pods (~15 eps/s with 15 pods, tested)
./sandbox/deploy.sh
python collect_distributed.py --env pusht --episodes 5000 --batch 15 --parallel 15
```

Output is HDF5 compatible with [`stable_worldmodel.data.HDF5Dataset`](https://github.com/galilai-group/stable-worldmodel).

## Architecture

```
┌──────────────────────────┐
│  Data Collection         │     ┌────────────────┐
│  collect_distributed.py  │────▶│ AKS Pods (N)   │──▶ trajectories
│  (local or AKS)          │◀────│ agent-sandbox   │
└──────────┬───────────────┘     └────────────────┘
           │ HDF5 / NPZ
           ▼
┌──────────────────────────┐     ┌────────────────┐
│  LeWM Training           │     │ GPU (T4/V100)  │
│  ViT encoder → z         │────▶│                │
│  Predictor → ẑ_next      │     │ MSE + λ·SIGReg │
│  SIGReg regularizer      │     └────────────────┘
└──────────┬───────────────┘
           │ trained model
           ▼
┌──────────────────────────┐
│  Evaluation              │
│  Rollout prediction      │
│  CEM planner (optional)  │
└──────────────────────────┘
```

## Project Structure

```
alpha-transformer/
├── experiment_lambda.py        # λ ablation experiment (the key finding)
├── train_lewm_gpu.py           # ViT pixel training (GPU-optimized)
├── train_lewm_pusht.py         # Push-T state-based training
├── collect_distributed.py      # Distributed data collection (local + AKS)
├── sweep_lambda.py             # Parallel λ sweep dispatcher
├── planner.py                  # CEM planner for evaluation
├── model/
│   ├── encoder.py              # MLP encoder + ViT encoder (HuggingFace)
│   ├── predictor.py            # Transformer predictor with AdaLN
│   └── sigreg.py               # SIGReg anti-collapse regularizer
├── sandbox/
│   ├── Dockerfile.gpu          # PyTorch CUDA training image
│   ├── Dockerfile              # Push-T data collection image
│   ├── gpu-train-job.yaml      # K8s Job: GPU training
│   ├── lambda-ablation-job.yaml# K8s Job: λ experiment
│   ├── sandbox-template.yaml   # K8s SandboxTemplate
│   ├── warm-pool.yaml          # Pre-warmed pod pool
│   └── deploy.sh               # One-command AKS deployment
└── results/                    # Saved models + datasets
```

## Related Repos

| Repo | What it does | How we use it |
|---|---|---|
| [le-wm](https://github.com/lucas-maes/le-wm) | LeWM paper code | Architecture reference (we reimplement, not fork) |
| [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | HDF5Dataset, environments, solvers | Data format compatibility |
| [stable-pretraining](https://github.com/galilai-group/stable-pretraining) | ViT backbones, training framework | Architecture reference for ViT config |
| [sandbox-arena](https://github.com/chokevin/sandbox-arena) | AKS sandbox platform | Distributed data collection infrastructure |
