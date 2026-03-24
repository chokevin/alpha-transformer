#!/usr/bin/env python3
"""Parallel hyperparameter sweep for LeWM using AKS sandbox pods.

Novel contribution: instead of sequential bisection search for λ (SIGReg
weight), dispatch N training runs in parallel across AKS pods. Each pod
trains a full LeWM model with a different λ value, evaluates it, and
returns the results. We pick the best.

The LeWM paper says λ is the only hyperparameter. They suggest bisection
(O(log n)). We do parallel grid search (O(1) wall-clock time).

Usage:
    # Local sweep (sequential)
    python sweep_lambda.py --local --n-values 8

    # Distributed sweep (all in parallel on AKS)
    python sweep_lambda.py --n-values 20 --parallel 20

    # Custom range
    python sweep_lambda.py --lambda-min 0.001 --lambda-max 10.0 --n-values 16
"""

import argparse
import json
import math
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


SWEEP_SCRIPT = '''
import json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LAMBDA = {lambd}
SEED = {seed}
EPOCHS = {epochs}
EMBED_DIM = {embed_dim}
SEQ_LEN = {seq_len}
GRID_SIZE = {grid_size}

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Minimal Snake env ---
class SnakeEnv:
    def __init__(self, gs=10):
        self.gs = gs
        self.reset()
    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        m = self.gs // 2
        self.snake = [(m, m)]
        self.food = self._pf()
        self.steps = 0
        self.mx = self.gs**2 * 2
        self.done = False
        self.score = 0
        return self._obs()
    def _pf(self):
        while True:
            p = (int(self.rng.integers(self.gs)), int(self.rng.integers(self.gs)))
            if p not in self.snake: return p
    def _obs(self):
        g = np.zeros((self.gs,self.gs,3), dtype=np.float32)
        for s in self.snake: g[s[0],s[1],0] = 1
        g[self.snake[0][0],self.snake[0][1],1] = 1
        g[self.food[0],self.food[1],2] = 1
        return g.flatten()
    def step(self, a):
        if self.done: return self._obs(), 0, True, {{"score": self.score}}
        self.steps += 1
        mv = [(-1,0),(0,1),(1,0),(0,-1)]
        dr,dc = mv[a]
        nr,nc = self.snake[0][0]+dr, self.snake[0][1]+dc
        if nr<0 or nr>=self.gs or nc<0 or nc>=self.gs or (nr,nc) in self.snake:
            self.done = True
            return self._obs(), -1.0, True, {{"score": self.score}}
        self.snake.insert(0, (nr,nc))
        if (nr,nc) == self.food:
            self.score += 1
            r = 1.0
            self.food = self._pf()
        else:
            self.snake.pop()
            r = -0.01
        if self.steps >= self.mx: self.done = True
        return self._obs(), r, self.done, {{"score": self.score}}

# --- BFS heuristic ---
from collections import deque
def bfs(s,g,bl,gs):
    q = deque([(s,[])])
    v = {{s}}
    mv = [(-1,0),(0,1),(1,0),(0,-1)]
    while q:
        (r,c),p = q.popleft()
        if (r,c)==g: return p
        for a,(dr,dc) in enumerate(mv):
            nr,nc = r+dr,c+dc
            if 0<=nr<gs and 0<=nc<gs and (nr,nc) not in v and (nr,nc) not in bl:
                v.add((nr,nc))
                q.append(((nr,nc),p+[a]))
    return None

def smart(obs, gs):
    g = obs.reshape(gs,gs,3)
    hp = np.argwhere(g[:,:,1]==1.0)
    fp = np.argwhere(g[:,:,2]==1.0)
    if len(hp)==0 or len(fp)==0: return np.random.randint(4)
    hr,hc = int(hp[0][0]),int(hp[0][1])
    fr,fc = int(fp[0][0]),int(fp[0][1])
    body = set(map(tuple, np.argwhere(g[:,:,0]==1.0)))
    mv = [(-1,0),(0,1),(1,0),(0,-1)]
    safe = [a for a,(dr,dc) in enumerate(mv) if 0<=hr+dr<gs and 0<=hc+dc<gs and (hr+dr,hc+dc) not in body]
    if not safe: return np.random.randint(4)
    if len(safe)==1: return safe[0]
    path = bfs((hr,hc),(fr,fc),body,gs)
    if path: return path[0]
    return safe[0]

# --- Collect data ---
trajs = []
for ep in range(200):
    env = SnakeEnv(GRID_SIZE)
    obs = env.reset(seed=ep+SEED*1000)
    ss, aa, rr = [], [], []
    while not env.done:
        ss.append(obs)
        a = smart(obs, GRID_SIZE) if np.random.random() > 0.05 else np.random.randint(4)
        aa.append(a)
        obs, r, d, info = env.step(a)
        rr.append(r)
    trajs.append({{"s": ss, "a": aa, "r": rr, "score": info["score"]}})

# --- Build dataset ---
obs_dim = GRID_SIZE**2 * 3
samples = []
for t in trajs:
    s = np.array(t["s"], dtype=np.float32)
    a = np.array(t["a"], dtype=np.int64)
    r = np.array(t["r"], dtype=np.float32)
    T = len(s)
    for st in range(0, max(1,T-SEQ_LEN+1), SEQ_LEN//2):
        e = min(st+SEQ_LEN, T)
        sl = e-st
        if sl < 3: continue
        sp = np.zeros((SEQ_LEN, obs_dim), dtype=np.float32)
        ap = np.zeros(SEQ_LEN, dtype=np.int64)
        sp[:sl] = s[st:e]
        ap[:sl] = a[st:e]
        samples.append((sp, ap))

# --- Model ---
class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(256, EMBED_DIM), nn.BatchNorm1d(EMBED_DIM))
    def forward(self, x):
        B,T,D = x.shape
        h = self.net(x.reshape(B*T, D))
        return self.proj(h).reshape(B, T, -1)

class Pred(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_emb = nn.Embedding(4, EMBED_DIM)
        self.inp = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.pos = nn.Embedding(SEQ_LEN, EMBED_DIM)
        layer = nn.TransformerEncoderLayer(EMBED_DIM, 4, EMBED_DIM*4, 0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.out = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.BatchNorm1d(EMBED_DIM))
    def forward(self, z, a):
        B,T,_ = z.shape
        h = self.inp(z) + self.act_emb(a) + self.pos(torch.arange(T,device=z.device))
        m = torch.triu(torch.ones(T,T,device=z.device),1).bool()
        h = self.tf(h, mask=m)
        return self.out(h.reshape(B*T,-1)).reshape(B,T,-1)

class SIGReg(nn.Module):
    def __init__(self):
        super().__init__()
        d = torch.randn(EMBED_DIM, 512)
        self.register_buffer("d", d / d.norm(dim=0, keepdim=True))
    def forward(self, z):
        z = (z - z.mean(0)) / (z.std(0) + 1e-8)
        h = z @ self.d
        cf_r = h.cos().mean(0)
        cf_i = h.sin().mean(0)
        return (cf_r**2 + cf_i**2 - math.exp(-1)).pow(2).mean()

enc = Enc()
pred = Pred()
sig = SIGReg()

# --- Train ---
opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=3e-4, weight_decay=1e-4)
t0 = time.time()

for epoch in range(EPOCHS):
    np.random.shuffle(samples)
    total_p, total_s, n = 0, 0, 0
    for i in range(0, len(samples)-32, 32):
        batch_s = torch.tensor(np.array([s[0] for s in samples[i:i+32]]))
        batch_a = torch.tensor(np.array([s[1] for s in samples[i:i+32]]))
        z = enc(batch_s)
        zp = pred(z, batch_a)
        pl = F.mse_loss(zp[:,:-1], z[:,1:])
        sl = sig(z.reshape(-1, EMBED_DIM))
        loss = pl + LAMBDA * sl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(pred.parameters()), 1.0)
        opt.step()
        total_p += pl.item()
        total_s += sl.item()
        n += 1

train_time = time.time() - t0

# --- Evaluate: rollout accuracy ---
enc.eval()
pred.eval()
errors = []
for i in range(min(50, len(samples))):
    s, a = torch.tensor(samples[i][0]).unsqueeze(0), torch.tensor(samples[i][1]).unsqueeze(0)
    with torch.no_grad():
        z = enc(s)
        zp = pred(z, a)
        err = (zp[:,:-1] - z[:,1:]).pow(2).mean().item()
        errors.append(err)

print(json.dumps({{
    "lambda": LAMBDA,
    "seed": SEED,
    "pred_loss": round(total_p / max(n,1), 6),
    "sigreg_loss": round(total_s / max(n,1), 6),
    "rollout_mse": round(float(np.mean(errors)), 6),
    "train_time": round(train_time, 1),
    "n_samples": len(samples),
}}))
'''


def run_sweep_local(lambda_values, args):
    """Run sweep sequentially on local machine."""
    results = []
    for i, lambd in enumerate(lambda_values):
        print(f"  [{i+1}/{len(lambda_values)}] λ={lambd:.4f}...")
        # Write script and run locally
        script = SWEEP_SCRIPT.format(
            lambd=lambd, seed=i, epochs=args.epochs,
            embed_dim=args.embed_dim, seq_len=args.seq_len,
            grid_size=args.grid_size,
        )
        import subprocess, sys, tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            f.flush()
            r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            result = json.loads(r.stdout.strip().split('\n')[-1])
            results.append(result)
            print(f"    pred={result['pred_loss']:.6f} sigreg={result['sigreg_loss']:.6f} "
                  f"rollout_mse={result['rollout_mse']:.6f} ({result['train_time']:.0f}s)")
        else:
            print(f"    FAILED: {r.stderr[:200]}")
    return results


def run_sweep_distributed(lambda_values, args):
    """Run sweep in parallel on AKS pods."""
    from k8s_agent_sandbox import SandboxClient

    results = []

    def run_one(lambd, idx):
        script = SWEEP_SCRIPT.format(
            lambd=lambd, seed=idx, epochs=args.epochs,
            embed_dim=args.embed_dim, seq_len=args.seq_len,
            grid_size=args.grid_size,
        )
        try:
            with SandboxClient(template_name=args.template, namespace=args.namespace) as sb:
                sb.write("sweep.py", script.encode())
                r = sb.run("python3 sweep.py", timeout=600)
            if r.exit_code == 0:
                return json.loads(r.stdout.strip().split('\n')[-1])
            else:
                print(f"    Pod error (λ={lambd:.4f}): {r.stderr[:100]}")
                return None
        except Exception as e:
            print(f"    Exception (λ={lambd:.4f}): {e}")
            return None

    print(f"  Dispatching {len(lambda_values)} training runs to {args.parallel} pods...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, lv, i): (i, lv) for i, lv in enumerate(lambda_values)}
        for future in as_completed(futures):
            idx, lv = futures[future]
            result = future.result()
            if result:
                results.append(result)
                print(f"    λ={lv:.4f}: rollout_mse={result['rollout_mse']:.6f}")

    elapsed = time.time() - t0
    print(f"  {len(results)}/{len(lambda_values)} completed in {elapsed:.0f}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="Parallel λ sweep for LeWM")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--n-values", type=int, default=8)
    parser.add_argument("--lambda-min", type=float, default=0.001)
    parser.add_argument("--lambda-max", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=20)
    parser.add_argument("--template", default="alpha-transformer-sandbox")
    parser.add_argument("--namespace", default="default")
    args = parser.parse_args()

    # Log-spaced λ values
    lambda_values = np.logspace(
        np.log10(args.lambda_min), np.log10(args.lambda_max), args.n_values
    ).tolist()

    print(f"LeWM λ Sweep: {args.n_values} values in [{args.lambda_min}, {args.lambda_max}]")
    print(f"Values: {[f'{v:.4f}' for v in lambda_values]}")
    print(f"Mode: {'local' if args.local else f'distributed ({args.parallel} pods)'}")

    if args.local:
        results = run_sweep_local(lambda_values, args)
    else:
        results = run_sweep_distributed(lambda_values, args)

    if not results:
        print("No results collected.")
        return

    # Find best λ by rollout MSE
    results.sort(key=lambda r: r["rollout_mse"])
    best = results[0]

    print(f"\n{'='*60}")
    print(f"SWEEP RESULTS ({len(results)} runs)")
    print(f"{'='*60}")
    print(f"{'λ':>10s} {'pred_loss':>12s} {'sigreg':>10s} {'rollout_mse':>12s}")
    print(f"{'-'*46}")
    for r in results:
        marker = " ★" if r == best else ""
        print(f"{r['lambda']:>10.4f} {r['pred_loss']:>12.6f} {r['sigreg_loss']:>10.6f} "
              f"{r['rollout_mse']:>12.6f}{marker}")
    print(f"\nBest λ = {best['lambda']:.4f} (rollout MSE = {best['rollout_mse']:.6f})")

    # Save results
    output = Path("results/lambda_sweep.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
