#!/usr/bin/env python3
"""Distributed RL training via sandbox-arena AKS pods.

Dispatches individual RL episodes to isolated AKS sandbox pods for
parallel execution. Requires:
  - AKS cluster with agent-sandbox controller
  - sandbox-arena K8s configs applied (kubectl apply -f ../sandbox-arena/sandbox/)
  - pip install k8s-agent-sandbox

Usage:
    # Distributed RL fine-tuning (50 parallel pods)
    python train_distributed.py --episodes 5000 --batch 50 --parallel 50

    # With custom template
    python train_distributed.py --episodes 5000 --template alpha-transformer-sandbox
"""

import argparse
import json
import time
import tempfile
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from model.transformer import TradingTransformer
from model.policy import save_checkpoint, load_checkpoint, build_context
from envs.features import STATE_DIM

# Episode script that runs inside each sandbox pod.
# It embeds the full environment + feature engineering + model inference.
EPISODE_SCRIPT_TEMPLATE = '''
import json
import math
import numpy as np
import sys

SEED = {seed}
TARGET_RETURN = {target_return}

# --- Commodity environment (self-contained for sandbox) ---
COMMODITIES = {{
    "gold": {{"mu": 0.06, "sigma": 0.15, "mean_rev": 0.02, "spread_bps": 5}},
    "oil": {{"mu": 0.03, "sigma": 0.35, "mean_rev": 0.05, "spread_bps": 10}},
    "wheat": {{"mu": 0.02, "sigma": 0.25, "mean_rev": 0.08, "spread_bps": 8}},
    "natgas": {{"mu": 0.01, "sigma": 0.45, "mean_rev": 0.10, "spread_bps": 15}},
}}
START_PRICES = {{"gold": 2000, "oil": 75, "wheat": 550, "natgas": 3.5}}
COMMODITY_NAMES = ["gold", "oil", "wheat", "natgas"]
N_DAYS = 252
INITIAL_CASH = 100000

def generate_prices(params, start_price, n_days, rng):
    dt = 1 / 252
    prices = np.zeros(n_days)
    prices[0] = start_price
    vol = params["sigma"]
    long_term = start_price
    for i in range(1, n_days):
        mr = params["mean_rev"] * (np.log(long_term) - np.log(prices[i-1])) * dt
        if i > 1:
            ret = np.log(prices[i-1] / prices[i-2])
            vol = 0.9 * vol + 0.1 * abs(ret) * np.sqrt(252)
            vol = np.clip(vol, params["sigma"] * 0.5, params["sigma"] * 2.0)
        drift = (params["mu"] - 0.5 * vol**2) * dt + mr
        shock = vol * np.sqrt(dt) * rng.standard_normal()
        prices[i] = prices[i-1] * np.exp(drift + shock)
    return prices

def compute_commodity_features(prices, idx):
    lookback = min(idx, 60)
    current = prices[idx]
    ret_1d = (prices[idx] / prices[max(0, idx-1)] - 1) if idx > 0 else 0.0
    ret_5d = (prices[idx] / prices[max(0, idx-5)] - 1) if idx >= 5 else 0.0
    ret_20d = (prices[idx] / prices[max(0, idx-20)] - 1) if idx >= 20 else 0.0
    ma5 = np.mean(prices[max(0, idx-4):idx+1])
    ma20 = np.mean(prices[max(0, idx-19):idx+1]) if idx >= 19 else ma5
    vol_window = prices[max(0, idx-19):idx+1]
    realized_vol = np.std(np.diff(np.log(vol_window))) * np.sqrt(252) if len(vol_window) > 1 else 0.0
    vol_regime = 0.0
    if idx >= 14:
        deltas = np.diff(prices[idx-14:idx+1])
        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
        losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-10
        rsi = (100 - 100 / (1 + gains / (losses + 1e-10))) / 100 - 0.5
    else:
        rsi = 0.0
    bb_window = prices[max(0, idx-19):idx+1]
    bb_pos = np.clip((current - np.mean(bb_window)) / (2 * np.std(bb_window) + 1e-10), -2, 2) / 2 if len(bb_window) > 1 else 0.0
    short_mom = ret_5d
    long_mom = ret_20d / 4 if idx >= 20 else ret_5d
    momentum_div = np.clip(short_mom - long_mom, -0.2, 0.2) * 5
    return np.array([
        np.clip(ret_1d, -0.2, 0.2) * 5, np.clip(ret_5d, -0.5, 0.5) * 2,
        np.clip(ret_20d, -1.0, 1.0), np.clip(current / ma5 - 1, -0.1, 0.1) * 10,
        np.clip(current / ma20 - 1, -0.2, 0.2) * 5, np.clip(realized_vol, 0, 1.0) * 2 - 1,
        vol_regime, rsi, bb_pos, momentum_div,
    ], dtype=np.float32)

def compute_cross_features(all_prices, idx, lookback=20):
    start = max(0, idx - lookback)
    returns = {{}}
    for name in COMMODITY_NAMES:
        p = all_prices[name][start:idx+1]
        returns[name] = np.diff(np.log(p)) if len(p) > 1 else np.array([0.0])
    corrs = []
    for i in range(4):
        for j in range(i+1, 4):
            r1, r2 = returns[COMMODITY_NAMES[i]], returns[COMMODITY_NAMES[j]]
            ml = min(len(r1), len(r2))
            corr = np.corrcoef(r1[-ml:], r2[-ml:])[0,1] if ml > 2 else 0.0
            corrs.append(0.0 if np.isnan(corr) else corr)
    return np.array(corrs, dtype=np.float32)

def get_obs(prices, day, cash, positions, prev_positions, portfolio_values, peak_value):
    feats = []
    for name in COMMODITY_NAMES:
        feats.append(compute_commodity_features(prices[name], day))
    cross = compute_cross_features(prices, day)
    pv = cash + sum(positions[n] * prices[n][day] for n in COMMODITY_NAMES)
    pv = max(pv, 1.0)
    weights = [abs(positions[n] * prices[n][day]) / pv for n in COMMODITY_NAMES]
    hhi = sum(w**2 for w in weights)
    turnover = -1.0
    if prev_positions:
        turnover = sum(abs(abs(positions[n]*prices[n][day]) - abs(prev_positions.get(n,0)*prices[n][day])) for n in COMMODITY_NAMES) / pv
        turnover = np.clip(turnover, 0, 1) * 2 - 1
    port = np.array([
        cash / pv, np.clip(pv / INITIAL_CASH - 1, -1, 5),
        np.clip((peak_value - pv) / peak_value, 0, 1) if peak_value > 0 else 0.0,
        (N_DAYS - day) / N_DAYS, hhi * 2 - 1, turnover
    ], dtype=np.float32)
    return np.concatenate(feats + [cross, port])

# --- Load model weights ---
import torch
import torch.nn as nn

class TradingTransformer(nn.Module):
    def __init__(self, state_dim=52, action_dim=4, hidden_dim=128,
                 n_heads=4, n_layers=4, max_seq_len=60, dropout=0.1):
        super().__init__()
        self.state_dim, self.hidden_dim = state_dim, hidden_dim
        self.max_seq_len, self.action_dim = max_seq_len, action_dim
        self.state_embed = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.action_embed = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.rtg_embed = nn.Sequential(nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim))
        self.pos_embed = nn.Embedding(max_seq_len * 3, hidden_dim)
        self.timestep_embed = nn.Embedding(max_seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embed_dropout = nn.Dropout(dropout)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, states, actions, rtgs, timesteps=None):
        B, T = states.shape[0], states.shape[1]
        device = states.device
        s_emb = self.state_embed(states)
        a_emb = self.action_embed(actions)
        r_emb = self.rtg_embed(rtgs)
        tokens = torch.zeros(B, T*3, self.hidden_dim, device=device)
        tokens[:, 0::3], tokens[:, 1::3], tokens[:, 2::3] = r_emb, s_emb, a_emb
        pos = torch.arange(T*3, device=device).unsqueeze(0)
        tokens = tokens + self.pos_embed(pos)
        if timesteps is None:
            timesteps = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        timesteps = timesteps.clamp(0, self.max_seq_len - 1)
        t_emb = self.timestep_embed(timesteps).unsqueeze(2).expand(-1,-1,3,-1).reshape(B, T*3, -1)
        tokens = self.embed_dropout(tokens + t_emb)
        mask = torch.triu(torch.ones(T*3, T*3, device=device), diagonal=1).bool()
        out = self.transformer(tokens, mask=mask)
        return self.action_head(out[:, 1::3])

checkpoint = torch.load("model.pt", weights_only=False)
config = checkpoint.get("config", {{}})
model = TradingTransformer(**config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --- Run episode ---
rng = np.random.default_rng(SEED)
prices = {{name: generate_prices(params, START_PRICES[name], N_DAYS, rng)
           for name, params in COMMODITIES.items()}}

day = 20
cash = INITIAL_CASH
positions = {{n: 0.0 for n in COMMODITY_NAMES}}
prev_positions = {{n: 0.0 for n in COMMODITY_NAMES}}
portfolio_values = [INITIAL_CASH]
peak_value = INITIAL_CASH
daily_returns = []

states_buf, actions_buf, rewards_buf = [], [], []
all_observations, all_actions, all_rewards = [], [], []

while day < N_DAYS - 1:
    obs = get_obs(prices, day, cash, positions, prev_positions, portfolio_values, peak_value)
    states_buf.append(obs)
    all_observations.append(obs.tolist())

    seq_len = min(len(states_buf), model.max_seq_len)
    s = torch.tensor(np.array(states_buf[-seq_len:]), dtype=torch.float32).unsqueeze(0)
    if actions_buf:
        a_len = min(len(actions_buf), seq_len)
        a = torch.tensor(np.array(actions_buf[-a_len:]), dtype=torch.float32).unsqueeze(0)
        if a.shape[1] < s.shape[1]:
            pad = torch.zeros(1, s.shape[1] - a.shape[1], 4)
            a = torch.cat([pad, a], dim=1)
    else:
        a = torch.zeros(1, seq_len, 4)
    current_rtg = (TARGET_RETURN - sum(rewards_buf)) / 100.0
    r = torch.full((1, seq_len, 1), current_rtg, dtype=torch.float32)

    with torch.no_grad():
        mu = model.forward(s, a, r)[:, -1]
    std = torch.exp(model.log_std.clamp(-2, 0.5))
    dist = torch.distributions.Normal(mu, std)
    action_t = torch.clamp(dist.sample(), -1, 1)
    log_prob = dist.log_prob(action_t).sum().item()

    action = action_t.squeeze(0).numpy()
    actions_buf.append(action)
    all_actions.append(action.tolist())

    # Execute trade
    pv_before = cash + sum(positions[n] * prices[n][day] for n in COMMODITY_NAMES)
    prev_positions = dict(positions)
    total_cost = 0
    for i, name in enumerate(COMMODITY_NAMES):
        target_weight = action[i] * 0.25
        target_value = pv_before * target_weight
        current_value = positions[name] * prices[name][day]
        trade_value = target_value - current_value
        if abs(trade_value) > 1:
            price = prices[name][day]
            spread = price * COMMODITIES[name]["spread_bps"] / 10000
            commission = abs(trade_value) * 2 / 10000
            units = trade_value / price
            positions[name] += units
            cash -= trade_value + np.sign(trade_value) * spread * abs(units)
            cash -= commission
            total_cost += commission + abs(units) * spread

    day += 1
    pv_after = cash + sum(positions[n] * prices[n][day] for n in COMMODITY_NAMES)
    portfolio_values.append(pv_after)
    peak_value = max(peak_value, pv_after)
    daily_return = (pv_after - pv_before) / pv_before if pv_before > 0 else 0.0
    daily_returns.append(daily_return)

    # Reward calculation
    drawdown = (peak_value - pv_after) / peak_value
    if len(daily_returns) > 5:
        recent_std = np.std(daily_returns[-20:]) + 1e-8
        sharpe_contribution = daily_return / recent_std
    else:
        sharpe_contribution = daily_return * 100
    weights = [abs(positions[n] * prices[n][day]) / pv_after if pv_after > 0 else 0 for n in COMMODITY_NAMES]
    hhi = sum(w**2 for w in weights)
    turnover = total_cost / pv_before if pv_before > 0 else 0
    reward = sharpe_contribution - drawdown * 5 - hhi * 2 - turnover * 50

    if day >= N_DAYS - 1:
        rets = np.array(daily_returns)
        sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252)
        max_dd = 0
        pk = portfolio_values[0]
        for v in portfolio_values:
            pk = max(pk, v)
            max_dd = max(max_dd, (pk - v) / pk)
        reward += sharpe * 15 - max_dd * 10

    rewards_buf.append(reward)
    all_rewards.append(reward)

total_return = (portfolio_values[-1] / INITIAL_CASH - 1) * 100
rets = np.array(daily_returns)
sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252)

print(json.dumps({{
    "observations": all_observations,
    "actions": all_actions,
    "rewards": all_rewards,
    "total_return": round(total_return, 2),
    "sharpe": round(sharpe, 3),
    "final_value": round(portfolio_values[-1], 2),
}}))
'''


def run_episode_in_sandbox(model_path, seed, template, namespace):
    """Run a single episode inside an AKS sandbox pod."""
    from k8s_agent_sandbox import SandboxClient

    script = EPISODE_SCRIPT_TEMPLATE.format(
        seed=seed,
        target_return=20.0,
    )

    with SandboxClient(template_name=template, namespace=namespace) as sandbox:
        # Upload model weights
        with open(model_path, "rb") as f:
            sandbox.write("model.pt", f.read())

        # Upload episode script
        sandbox.write("episode.py", script.encode())

        # Execute
        result = sandbox.run("python3 episode.py", timeout=120)

    if result.exit_code != 0:
        print(f"  Sandbox error (seed={seed}): {result.stderr[:200]}")
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  JSON decode error (seed={seed})")
        return None


def main():
    parser = argparse.ArgumentParser(description="Distributed RL training via AKS")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=50,
                        help="Episodes per policy update")
    parser.add_argument("--parallel", type=int, default=50,
                        help="Concurrent sandbox pods")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--template", default="arena-sandbox",
                        help="K8s SandboxTemplate name")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--model-path", default="results/trading_transformer.pt")
    parser.add_argument("--eval-interval", type=int, default=200)
    args = parser.parse_args()

    # Load pretrained model
    if not Path(args.model_path).exists():
        print(f"No model at {args.model_path}. Run: python train.py --phase all")
        return

    model, metadata = load_checkpoint(args.model_path, TradingTransformer)
    print(f"Loaded model: {model.param_count():,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_sharpe = -float("inf")
    best_state = None
    tmp_model = Path(tempfile.mkdtemp()) / "policy.pt"

    for ep in range(0, args.episodes, args.batch):
        model.train()
        save_checkpoint(model, tmp_model)

        # Dispatch episodes to sandbox pods in parallel
        print(f"\n  Dispatching batch {ep}-{ep+args.batch} to {args.parallel} pods...")
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(run_episode_in_sandbox, str(tmp_model),
                            seed=ep + i, template=args.template,
                            namespace=args.namespace): i
                for i in range(args.batch)
            }
            trajectories = []
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    trajectories.append(result)

        elapsed = time.time() - t0
        print(f"  Received {len(trajectories)}/{args.batch} trajectories in {elapsed:.1f}s")

        if len(trajectories) < 2:
            print("  Too few valid trajectories, skipping update")
            continue

        # REINFORCE update
        all_returns = []
        all_log_probs = []

        for traj in trajectories:
            rewards = traj["rewards"]
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            all_returns.extend(returns)

            # Re-compute log probs locally
            obs_list = traj["observations"]
            act_list = traj["actions"]
            states_buf = []
            actions_buf = []
            rewards_buf = []

            for t_idx in range(len(obs_list)):
                obs = np.array(obs_list[t_idx])
                states_buf.append(obs)

                s, a, r = build_context(states_buf, actions_buf, rewards_buf, model)
                with torch.no_grad():
                    mu = model.forward(s, a, r)[:, -1]

                std = torch.exp(model.log_std.clamp(-2, 0.5))
                dist = torch.distributions.Normal(mu, std)
                action_t = torch.tensor(act_list[t_idx], dtype=torch.float32).unsqueeze(0)
                log_prob = dist.log_prob(action_t).sum()
                all_log_probs.append(log_prob)

                actions_buf.append(np.array(act_list[t_idx]))
                if t_idx < len(rewards):
                    rewards_buf.append(rewards[t_idx])

        returns_t = torch.tensor(all_returns, dtype=torch.float32)
        advantages = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        clipped_adv = torch.clamp(advantages, -2, 2)
        log_probs_t = torch.stack(all_log_probs)

        loss = -(log_probs_t * clipped_adv).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # Evaluate
        traj_returns = [t["total_return"] for t in trajectories]
        traj_sharpes = [t["sharpe"] for t in trajectories]
        avg_ret = np.mean(traj_returns)
        avg_sharpe = np.mean(traj_sharpes)

        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ★"
        else:
            marker = ""

        print(f"  Ep {ep+args.batch:>5d} | ret={avg_ret:>+.2f}% | "
              f"sharpe={avg_sharpe:>+.3f} | loss={loss.item():.4f}{marker}")

        history.append({
            "episode": ep + args.batch,
            "avg_return": round(avg_ret, 2),
            "avg_sharpe": round(avg_sharpe, 3),
            "loss": round(loss.item(), 4),
            "n_valid": len(trajectories),
        })

    # Restore best and save
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model (Sharpe={best_sharpe:.3f})")

    save_checkpoint(model, args.model_path, metadata={
        "phase": "distributed_rl",
        "episodes": args.episodes,
        "best_sharpe": best_sharpe,
    })
    print(f"Saved to {args.model_path}")

    history_path = Path(f"results/distributed_rl_{int(time.time())}.json")
    with open(history_path, "w") as f:
        json.dump({"history": history}, f, indent=2)

    # Cleanup
    if tmp_model.exists():
        tmp_model.unlink()


if __name__ == "__main__":
    main()
