"""Snake game Gym environment.

Classic Snake game suitable for Decision Transformer training.

Observation: 300-dim (10×10 grid × 3 channels: body, head, food)
Action: Discrete(4) — 0=up, 1=right, 2=down, 3=left
Reward: +1 food, -1 death, -0.01 per step
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class SnakeEnv(gym.Env):
    """Snake game environment.

    Observation: flattened grid (grid_size² × 3 channels)
      Channel 0: snake body, Channel 1: snake head, Channel 2: food
    Action: 0=up, 1=right, 2=down, 3=left
    Reward: +1 food, -1 death, -0.01 per step
    """

    metadata = {"render_modes": []}

    def __init__(self, grid_size: int = 10):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size * grid_size * 3,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        mid = self.grid_size // 2
        self.snake = [(mid, mid)]
        self.direction = 1
        self.food = self._place_food()
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 2
        self.done = False
        self.score = 0
        return self._get_obs(), {}

    def _place_food(self):
        while True:
            pos = (int(self.rng.integers(self.grid_size)),
                   int(self.rng.integers(self.grid_size)))
            if pos not in self.snake:
                return pos

    def _get_obs(self):
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for seg in self.snake:
            grid[seg[0], seg[1], 0] = 1.0
        head = self.snake[0]
        grid[head[0], head[1], 1] = 1.0
        grid[self.food[0], self.food[1], 2] = 1.0
        return grid.flatten()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        self.steps += 1
        head = self.snake[0]
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        dr, dc = moves[action]
        new_head = (head[0] + dr, head[1] + dc)

        # Wall or self collision = death
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
                new_head[1] < 0 or new_head[1] >= self.grid_size or
                new_head in self.snake):
            self.done = True
            return self._get_obs(), -1.0, True, False, {"score": self.score}

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            reward = 1.0
            self.food = self._place_food()
        else:
            self.snake.pop()
            reward = -0.01

        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, False, {"score": self.score}
