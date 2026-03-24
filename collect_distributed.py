#!/usr/bin/env python3
"""Distributed data collection via AKS sandbox pods.

Collects trajectories from any Gym environment by dispatching episodes
to isolated AKS sandbox pods in parallel. Produces HDF5 datasets
compatible with the LeWM training pipeline (stable-worldmodel format).

This is the value-add over the official le-wm repo: they collect data
locally, we parallelize across 50+ pods for 10-50x speedup on
expensive environments (3D rendering, complex physics, long episodes).

Architecture:
  ┌──────────────────────┐
  │  Collector (local)   │
  │  - Dispatches N pods │     ┌──────────┐
  │  - Collects JSON     │────▶│ Pod 1    │──▶ trajectory JSON
  │  - Saves HDF5        │     │ Pod 2    │──▶ trajectory JSON
  │                      │     │ ...      │
  │                      │◀────│ Pod N    │──▶ trajectory JSON
  └──────────────────────┘     └──────────┘

Usage:
    # Collect Snake trajectories (local, no AKS)
    python collect_distributed.py --env snake --episodes 5000 --local

    # Collect via AKS sandbox pods
    python collect_distributed.py --env snake --episodes 10000 --batch 50 --parallel 50

    # Collect Push-T pixel data (slow locally, fast on AKS)
    python collect_distributed.py --env pusht --episodes 5000 --batch 50 --parallel 50

    # Custom environment script
    python collect_distributed.py --script my_env.py --episodes 1000 --batch 50
"""

import argparse
import json
import time
import h5py
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# Environment episode scripts (embedded in sandbox pods)
# ============================================================

SNAKE_EPISODE_SCRIPT = '''
import json
import numpy as np
from collections import deque

SEED = {seed}
GRID_SIZE = {grid_size}

class SnakeEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()
    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        mid = self.grid_size // 2
        self.snake = [(mid, mid)]
        self.food = self._place_food()
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 2
        self.done = False
        self.score = 0
        return self._get_obs()
    def _place_food(self):
        while True:
            pos = (int(self.rng.integers(self.grid_size)), int(self.rng.integers(self.grid_size)))
            if pos not in self.snake:
                return pos
    def _get_obs(self):
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for seg in self.snake:
            grid[seg[0], seg[1], 0] = 1.0
        grid[self.snake[0][0], self.snake[0][1], 1] = 1.0
        grid[self.food[0], self.food[1], 2] = 1.0
        return grid.flatten()
    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, {{"score": self.score}}
        self.steps += 1
        moves = [(-1,0),(0,1),(1,0),(0,-1)]
        dr, dc = moves[action]
        nr, nc = self.snake[0][0]+dr, self.snake[0][1]+dc
        if nr<0 or nr>=self.grid_size or nc<0 or nc>=self.grid_size or (nr,nc) in self.snake:
            self.done = True
            return self._get_obs(), -1.0, True, {{"score": self.score}}
        self.snake.insert(0, (nr, nc))
        if (nr, nc) == self.food:
            self.score += 1
            reward = 1.0
            self.food = self._place_food()
        else:
            self.snake.pop()
            reward = -0.01
        if self.steps >= self.max_steps:
            self.done = True
        return self._get_obs(), reward, self.done, {{"score": self.score}}

def flood_fill(start, blocked, gs):
    visited = set()
    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r,c) in visited or (r,c) in blocked or r<0 or r>=gs or c<0 or c>=gs:
            continue
        visited.add((r,c))
        stack.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
    return len(visited)

def bfs_path(start, goal, blocked, gs):
    queue = deque([(start, [])])
    visited = {{start}}
    moves = [(-1,0),(0,1),(1,0),(0,-1)]
    while queue:
        (r,c), path = queue.popleft()
        if (r,c) == goal:
            return path
        for a,(dr,dc) in enumerate(moves):
            nr, nc = r+dr, c+dc
            if 0<=nr<gs and 0<=nc<gs and (nr,nc) not in visited and (nr,nc) not in blocked:
                visited.add((nr,nc))
                queue.append(((nr,nc), path+[a]))
    return None

def smart_action(obs, gs):
    grid = obs.reshape(gs, gs, 3)
    hp = np.argwhere(grid[:,:,1]==1.0)
    fp = np.argwhere(grid[:,:,2]==1.0)
    if len(hp)==0 or len(fp)==0:
        return np.random.randint(4)
    hr, hc = int(hp[0][0]), int(hp[0][1])
    fr, fc = int(fp[0][0]), int(fp[0][1])
    body = set(map(tuple, np.argwhere(grid[:,:,0]==1.0)))
    moves = [(-1,0),(0,1),(1,0),(0,-1)]
    safe = [a for a,(dr,dc) in enumerate(moves)
            if 0<=hr+dr<gs and 0<=hc+dc<gs and (hr+dr,hc+dc) not in body]
    if not safe: return np.random.randint(4)
    if len(safe)==1: return safe[0]
    path = bfs_path((hr,hc),(fr,fc), body, gs)
    if path:
        a = path[0]
        dr,dc = moves[a]
        nr,nc = hr+dr, hc+dc
        nb = body | {{(nr,nc)}}
        if flood_fill((nr,nc), nb-{{(nr,nc)}}, gs) >= len(body):
            return a
    best_a, best_r = safe[0], -1
    for a in safe:
        dr,dc = moves[a]
        nr,nc = hr+dr, hc+dc
        nb = body | {{(nr,nc)}}
        r = flood_fill((nr,nc), nb-{{(nr,nc)}}, gs)
        if r > best_r: best_r, best_a = r, a
    return best_a

rng = np.random.default_rng(SEED)
noise = rng.choice([0.02, 0.05, 0.1, 0.15], p=[0.3, 0.3, 0.25, 0.15])
env = SnakeEnv(GRID_SIZE)
obs = env.reset(seed=SEED)
states, actions, rewards = [], [], []
while not env.done:
    states.append(obs.tolist())
    if rng.random() < noise:
        grid = obs.reshape(GRID_SIZE, GRID_SIZE, 3)
        hp = np.argwhere(grid[:,:,1]==1.0)
        body = set(map(tuple, np.argwhere(grid[:,:,0]==1.0)))
        mvs = [(-1,0),(0,1),(1,0),(0,-1)]
        safe = [a for a,(dr,dc) in enumerate(mvs)
                if len(hp)>0 and 0<=hp[0][0]+dr<GRID_SIZE and 0<=hp[0][1]+dc<GRID_SIZE
                and (hp[0][0]+dr,hp[0][1]+dc) not in body]
        action = int(rng.choice(safe)) if safe else int(rng.integers(4))
    else:
        action = int(smart_action(obs, GRID_SIZE))
    actions.append(action)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)

print(json.dumps({{
    "states": states, "actions": actions, "rewards": rewards,
    "score": info.get("score", 0), "length": len(states),
}}))
'''


# ============================================================
# Local collection (no AKS)
# ============================================================

def collect_local(env_name, n_episodes, grid_size=10):
    """Collect trajectories locally using Python multiprocessing."""
    if env_name == "snake":
        from envs.snake import SnakeEnv
        from train_lewm_snake import _smart_snake

        trajectories = []
        for ep in range(n_episodes):
            env = SnakeEnv(grid_size=grid_size)
            obs, _ = env.reset()
            states, actions, rewards = [], [], []
            noise = np.random.choice([0.02, 0.05, 0.1, 0.15], p=[0.3, 0.3, 0.25, 0.15])
            done = False
            while not done:
                states.append(obs)
                if np.random.random() < noise:
                    grid = obs.reshape(grid_size, grid_size, 3)
                    head_pos = np.argwhere(grid[:, :, 1] == 1.0)
                    body = set(map(tuple, np.argwhere(grid[:, :, 0] == 1.0)))
                    mvs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    if len(head_pos) > 0:
                        hr, hc = head_pos[0]
                        safe = [a for a, (dr, dc) in enumerate(mvs)
                                if 0 <= hr+dr < grid_size and 0 <= hc+dc < grid_size
                                and (hr+dr, hc+dc) not in body]
                        action = int(np.random.choice(safe)) if safe else int(np.random.randint(4))
                    else:
                        action = int(np.random.randint(4))
                else:
                    action = int(_smart_snake(obs, grid_size))
                actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                done = terminated or truncated

            trajectories.append({
                "states": [s.tolist() for s in states],
                "actions": actions,
                "rewards": rewards,
                "score": info.get("score", 0),
                "length": len(states),
            })

            if (ep + 1) % 500 == 0:
                recent = trajectories[-500:]
                scores = [t["score"] for t in recent]
                print(f"  [{ep+1}/{n_episodes}] avg={np.mean(scores):.1f} max={max(scores)}")

        return trajectories
    else:
        raise ValueError(f"Unknown env: {env_name}. Supported: snake")


# ============================================================
# Distributed collection (AKS sandbox pods)
# ============================================================

def collect_distributed(env_name, n_episodes, batch_size, parallel,
                        template, namespace, grid_size=10):
    """Collect trajectories via AKS sandbox pods."""
    from k8s_agent_sandbox import SandboxClient

    if env_name == "snake":
        script_template = SNAKE_EPISODE_SCRIPT
    else:
        raise ValueError(f"Unknown env: {env_name}")

    trajectories = []
    n_batches = (n_episodes + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_episodes)
        current_batch = end - start

        t0 = time.time()
        print(f"  Batch {batch_idx+1}/{n_batches}: dispatching {current_batch} episodes to {parallel} pods...")

        def run_one(seed):
            script = script_template.format(seed=seed, grid_size=grid_size)
            try:
                with SandboxClient(template_name=template, namespace=namespace) as sandbox:
                    sandbox.write("episode.py", script.encode())
                    result = sandbox.run("python3 episode.py", timeout=120)
                if result.exit_code == 0:
                    return json.loads(result.stdout.strip().split("\n")[-1])
                else:
                    print(f"    Pod error (seed={seed}): {result.stderr[:100]}")
                    return None
            except Exception as e:
                print(f"    Exception (seed={seed}): {e}")
                return None

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(run_one, start + i): i for i in range(current_batch)}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    trajectories.append(result)

        elapsed = time.time() - t0
        valid = len(trajectories) - (batch_idx * batch_size if batch_idx > 0 else 0)
        print(f"    Got {valid}/{current_batch} in {elapsed:.1f}s "
              f"({current_batch/elapsed:.0f} eps/s)")

    return trajectories


# ============================================================
# Save as HDF5 (compatible with stable-worldmodel)
# ============================================================

def save_hdf5(trajectories, output_path, obs_key="pixels"):
    """Save trajectories as HDF5 dataset.

    Format compatible with stable-worldmodel's HDF5Dataset:
      - pixels: (N, obs_dim) or (N, C, H, W) observations
      - action: (N, action_dim) actions
      - reward: (N,) rewards
      - episode_idx: (N,) which episode each row belongs to
      - step_idx: (N,) timestep within episode
    """
    # Flatten all trajectories into rows
    all_obs, all_actions, all_rewards = [], [], []
    all_episode_idx, all_step_idx = [], []

    for ep_idx, traj in enumerate(trajectories):
        states = np.array(traj["states"], dtype=np.float32)
        actions = np.array(traj["actions"])
        rewards = np.array(traj["rewards"], dtype=np.float32)
        T = len(states)

        all_obs.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_episode_idx.append(np.full(T, ep_idx, dtype=np.int32))
        all_step_idx.append(np.arange(T, dtype=np.int32))

    obs = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    episode_idx = np.concatenate(all_episode_idx, axis=0)
    step_idx = np.concatenate(all_step_idx, axis=0)

    # Handle action shape
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)  # (N, 1) for discrete

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset(obs_key, data=obs, compression="gzip", compression_opts=4)
        f.create_dataset("action", data=actions.astype(np.float32), compression="gzip")
        f.create_dataset("reward", data=rewards, compression="gzip")
        f.create_dataset("episode_idx", data=episode_idx, compression="gzip")
        f.create_dataset("step_idx", data=step_idx, compression="gzip")

    n_episodes = len(trajectories)
    n_rows = len(obs)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  Saved: {output_path}")
    print(f"  {n_episodes} episodes, {n_rows} rows, {size_mb:.1f} MB")


def save_json(trajectories, output_path):
    """Save as JSON (for our training scripts)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectories, f)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Distributed data collection for LeWM training"
    )
    parser.add_argument("--env", default="snake", choices=["snake"],
                        help="Environment to collect from")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--min-score-percentile", type=int, default=0,
                        help="Filter: keep only top N%% trajectories (0=keep all)")

    # Local vs distributed
    parser.add_argument("--local", action="store_true",
                        help="Collect locally (no AKS)")
    parser.add_argument("--batch", type=int, default=50,
                        help="Episodes per batch (distributed only)")
    parser.add_argument("--parallel", type=int, default=50,
                        help="Concurrent pods (distributed only)")
    parser.add_argument("--template", default="arena-sandbox",
                        help="K8s SandboxTemplate name")
    parser.add_argument("--namespace", default="default")

    # Output
    parser.add_argument("--output", default=None,
                        help="Output path (auto-generated if not set)")
    parser.add_argument("--format", default="both", choices=["json", "hdf5", "both"])

    args = parser.parse_args()

    print(f"Collecting {args.episodes} episodes from {args.env}")
    print(f"Mode: {'local' if args.local else f'distributed ({args.parallel} pods)'}")
    t0 = time.time()

    if args.local:
        trajectories = collect_local(args.env, args.episodes, args.grid_size)
    else:
        trajectories = collect_distributed(
            args.env, args.episodes, args.batch, args.parallel,
            args.template, args.namespace, args.grid_size,
        )

    elapsed = time.time() - t0
    scores = [t.get("score", 0) for t in trajectories]
    print(f"\nCollected {len(trajectories)} trajectories in {elapsed:.1f}s "
          f"({len(trajectories)/elapsed:.0f} eps/s)")
    print(f"Scores: avg={np.mean(scores):.1f}, p50={np.median(scores):.0f}, "
          f"p90={np.percentile(scores,90):.0f}, max={max(scores)}")

    # Filter
    if args.min_score_percentile > 0:
        threshold = np.percentile(scores, args.min_score_percentile)
        before = len(trajectories)
        trajectories = [t for t in trajectories if t.get("score", 0) >= threshold]
        print(f"Filtered: {len(trajectories)}/{before} (threshold={threshold})")

    # Save
    base = args.output or f"results/{args.env}_data_{len(trajectories)}"
    if args.format in ("json", "both"):
        save_json(trajectories, f"{base}.json")
    if args.format in ("hdf5", "both"):
        save_hdf5(trajectories, f"{base}.h5")


if __name__ == "__main__":
    main()
