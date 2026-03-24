"""Commodity trading Gym environment with realistic market dynamics.

Simulates trading Gold, Oil, Wheat, and Natural Gas with:
- Historical-like price patterns (trend, mean-reversion, volatility clustering)
- Transaction costs (spread + commission)
- Position sizing (fractional, not just all-in)
- Multiple technical indicators as observations
- Portfolio management across multiple commodities

This is a proper trading environment suitable for RL training.
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


# Realistic commodity parameters (annualized)
COMMODITIES = {
    "gold": {"mu": 0.06, "sigma": 0.15, "mean_rev": 0.02, "spread_bps": 5},
    "oil": {"mu": 0.03, "sigma": 0.35, "mean_rev": 0.05, "spread_bps": 10},
    "wheat": {"mu": 0.02, "sigma": 0.25, "mean_rev": 0.08, "spread_bps": 8},
    "natgas": {"mu": 0.01, "sigma": 0.45, "mean_rev": 0.10, "spread_bps": 15},
}

# Starting prices (approximate real-world)
START_PRICES = {"gold": 2000, "oil": 75, "wheat": 550, "natgas": 3.5}


def generate_prices(params: dict, start_price: float, n_days: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Generate realistic commodity price series.

    Uses a mean-reverting GBM with volatility clustering (GARCH-like).
    """
    dt = 1 / 252  # daily
    prices = np.zeros(n_days)
    prices[0] = start_price
    vol = params["sigma"]
    long_term = start_price

    for i in range(1, n_days):
        # Mean reversion toward long-term price
        mr = params["mean_rev"] * (np.log(long_term) - np.log(prices[i-1])) * dt

        # Volatility clustering (simple GARCH-like)
        if i > 1:
            ret = np.log(prices[i-1] / prices[i-2])
            vol = 0.9 * vol + 0.1 * abs(ret) * np.sqrt(252)
            vol = np.clip(vol, params["sigma"] * 0.5, params["sigma"] * 2.0)

        # GBM step
        drift = (params["mu"] - 0.5 * vol**2) * dt + mr
        shock = vol * np.sqrt(dt) * rng.standard_normal()
        prices[i] = prices[i-1] * np.exp(drift + shock)

    return prices


def compute_features(prices: np.ndarray, idx: int) -> np.ndarray:
    """Compute technical features for a single commodity at time idx."""
    if idx < 20:
        lookback = max(idx, 1)
    else:
        lookback = 20

    window = prices[max(0, idx-lookback+1):idx+1]
    current = prices[idx]

    # Returns
    ret_1d = (prices[idx] / prices[max(0, idx-1)] - 1) if idx > 0 else 0
    ret_5d = (prices[idx] / prices[max(0, idx-5)] - 1) if idx >= 5 else 0
    ret_20d = (prices[idx] / prices[max(0, idx-20)] - 1) if idx >= 20 else 0

    # Moving averages
    ma5 = np.mean(prices[max(0, idx-4):idx+1])
    ma20 = np.mean(window)

    # Volatility (annualized)
    if len(window) > 1:
        log_rets = np.diff(np.log(window))
        vol = np.std(log_rets) * np.sqrt(252) if len(log_rets) > 0 else 0
    else:
        vol = 0

    # RSI (14-day)
    if idx >= 14:
        deltas = np.diff(prices[idx-14:idx+1])
        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gains / (losses + 1e-10)
        rsi = 100 - 100 / (1 + rs)
    else:
        rsi = 50

    # Bollinger band position (-1 to 1)
    if len(window) > 1:
        std = np.std(window)
        bb_pos = (current - ma20) / (2 * std + 1e-10)
    else:
        bb_pos = 0

    return np.array([
        ret_1d, ret_5d, ret_20d,         # momentum
        current / ma5 - 1,               # price vs MA5
        current / ma20 - 1,              # price vs MA20
        vol,                              # realized volatility
        rsi / 100 - 0.5,                 # RSI normalized to [-0.5, 0.5]
        np.clip(bb_pos, -2, 2) / 2,      # Bollinger position normalized
    ], dtype=np.float32)


class CommodityTradingEnv(gym.Env):
    """Multi-commodity trading environment.

    Observation (per commodity × 8 features + portfolio state):
        Per commodity: [ret_1d, ret_5d, ret_20d, price_vs_ma5, price_vs_ma20, vol, rsi, bb]
        Portfolio: [cash_pct, total_return, drawdown, days_remaining]

    Action (per commodity): continuous [-1, 1]
        -1 = full short, 0 = no position, 1 = full long
        The action represents target position as fraction of portfolio.

    Reward: daily Sharpe-like return (return / volatility)
    """

    metadata = {"render_modes": []}

    def __init__(self, n_days: int = 252, initial_cash: float = 100_000,
                 commission_bps: float = 2):
        super().__init__()

        self.n_commodities = len(COMMODITIES)
        self.commodity_names = list(COMMODITIES.keys())
        self.n_days = n_days
        self.initial_cash = initial_cash
        self.commission_bps = commission_bps

        # Features per commodity (8) + portfolio state (4) = 36 total
        obs_dim = self.n_commodities * 8 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: target position for each commodity [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_commodities,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Generate price series for each commodity
        self.prices = {}
        for name, params in COMMODITIES.items():
            self.prices[name] = generate_prices(
                params, START_PRICES[name], self.n_days, self.rng
            )

        self.day = 20  # start after enough history for features
        self.cash = self.initial_cash
        self.positions = {name: 0.0 for name in self.commodity_names}  # units held
        self.portfolio_values = [self.initial_cash]
        self.peak_value = self.initial_cash

        return self._get_obs(), {}

    def _portfolio_value(self):
        val = self.cash
        for name in self.commodity_names:
            val += self.positions[name] * self.prices[name][self.day]
        return val

    def _get_obs(self):
        features = []
        for name in self.commodity_names:
            feat = compute_features(self.prices[name], self.day)
            features.append(feat)

        pv = self._portfolio_value()
        total_return = (pv / self.initial_cash) - 1
        drawdown = (self.peak_value - pv) / self.peak_value if self.peak_value > 0 else 0
        days_remaining = (self.n_days - self.day) / self.n_days

        portfolio_state = np.array([
            self.cash / pv if pv > 0 else 1.0,  # cash percentage
            np.clip(total_return, -1, 5),        # total return
            np.clip(drawdown, 0, 1),             # current drawdown
            days_remaining,                       # time remaining
        ], dtype=np.float32)

        return np.concatenate(features + [portfolio_state])

    def step(self, action):
        action = np.clip(action, -1, 1)
        pv_before = self._portfolio_value()

        # Rebalance to target positions
        total_cost = 0
        for i, name in enumerate(self.commodity_names):
            target_weight = action[i] * 0.25  # max 25% per commodity
            target_value = pv_before * target_weight
            current_value = self.positions[name] * self.prices[name][self.day]
            trade_value = target_value - current_value

            if abs(trade_value) > 1:  # minimum trade size
                price = self.prices[name][self.day]
                spread = price * COMMODITIES[name]["spread_bps"] / 10000
                commission = abs(trade_value) * self.commission_bps / 10000

                # Execute trade
                units = trade_value / price
                self.positions[name] += units
                self.cash -= trade_value + np.sign(trade_value) * spread * abs(units)
                self.cash -= commission
                total_cost += commission + abs(units) * spread

        # Advance day
        self.day += 1
        done = self.day >= self.n_days - 1

        # Calculate reward
        pv_after = self._portfolio_value()
        self.portfolio_values.append(pv_after)
        self.peak_value = max(self.peak_value, pv_after)

        daily_return = (pv_after - pv_before) / pv_before if pv_before > 0 else 0

        # Reward: risk-adjusted return (penalize drawdowns)
        drawdown = (self.peak_value - pv_after) / self.peak_value
        reward = daily_return * 100 - drawdown * 10  # scale for RL

        info = {
            "portfolio_value": round(pv_after, 2),
            "total_return": round((pv_after / self.initial_cash - 1) * 100, 2),
            "drawdown": round(drawdown * 100, 2),
            "cash": round(self.cash, 2),
            "positions": {n: round(self.positions[n], 4) for n in self.commodity_names},
            "trade_cost": round(total_cost, 2),
            "day": self.day,
        }

        if done:
            # Final stats
            returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            info["sharpe"] = round(sharpe, 3)
            info["final_value"] = round(pv_after, 2)
            # Bonus for good Sharpe
            reward += sharpe * 10

        return self._get_obs(), reward, done, False, info
