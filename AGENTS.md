# alpha-transformer Agent Instructions

## What This Repo Does
LeWM (Le World Model) reimplementation with ViT pixel training.
Key finding: SIGReg λ=0.01–0.032 beats the paper's default λ=0.1 by 29%.

## Training

```bash
# GPU training (local or AKS)
python train_lewm_gpu.py --mode pixel --episodes 200 --epochs 50 --lambd 0.032

# λ ablation experiment
python experiment_lambda.py --episodes 200 --lambdas 0.01,0.032,0.1,0.32

# State-based Push-T training
python train_lewm_pusht.py --phase all --episodes 500
```

## GPU Training on AKS

```bash
# Build image via ACR
az acr build --subscription "AKS Airlock" --registry suliregistry \
  --image alpha-train:latest -f sandbox/Dockerfile.gpu --platform linux/amd64 .

# Submit job
kubectl apply -f sandbox/gpu-train-job.yaml        # single training
kubectl apply -f sandbox/lambda-ablation-job.yaml   # λ ablation
```

## Distributed Data Collection

```bash
# Local
python collect_distributed.py --env pusht --episodes 500 --local

# AKS pods (deploy infra first: ./sandbox/deploy.sh)
python collect_distributed.py --env pusht --episodes 5000 --batch 15 --parallel 15
```

## Key Files
- `experiment_lambda.py` — λ ablation (the novel finding)
- `train_lewm_gpu.py` — ViT pixel training, GPU-optimized
- `model/encoder.py` — MLP + ViT encoders
- `model/predictor.py` — Transformer predictor with AdaLN
- `model/sigreg.py` — SIGReg regularizer
