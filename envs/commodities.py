"""Commodity trading Gym environment with realistic market dynamics.

Simulates trading Gold, Oil, Wheat, and Natural Gas with:
- Historical-like price patterns (trend, mean-reversion, volatility clustering)
- Transaction costs (spread + commission)
- Position sizing (fractional, not just all-in)
- Enhanced technical indicators + cross-commodity features
- Sharpe-based reward shaping with concentration/turnover penalties
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from envs.features import (
    compute_commodity_features,
    compute_cross_commodity_features,
    compute_portfolio_features,
    STATE_DIM,
)


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


class CommodityTradingEnv(gym.Env):
    """Multi-commodity trading environment with enhanced features and reward shaping.

    Observation (52-dim):
        Per commodity (10 × 4 = 40): returns, MA ratios, vol, vol regime, RSI, BB, momentum div
        Cross-commodity (6): pairwise rolling correlations
        Portfolio (6): cash, return, drawdown, time, concentration, turnover

    Action (4-dim): continuous [-1, 1] target position per commodity

    Reward: Sharpe-based with concentration/turnover penalties
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

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_commodities,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.prices = {}
        for name, params in COMMODITIES.items():
            self.prices[name] = generate_prices(
                params, START_PRICES[name], self.n_days, self.rng
            )

        self.day = 20  # start after enough history for features
        self.cash = self.initial_cash
        self.positions = {name: 0.0 for name in self.commodity_names}
        self.prev_positions = {name: 0.0 for name in self.commodity_names}
        self.portfolio_values = [self.initial_cash]
        self.peak_value = self.initial_cash
        self.daily_returns = []

        return self._get_obs(), {}

    def _portfolio_value(self):
        val = self.cash
        for name in self.commodity_names:
            val += self.positions[name] * self.prices[name][self.day]
        return val

    def _get_obs(self):
        # Per-commodity features (10 each)
        commodity_feats = []
        for name in self.commodity_names:
            feat = compute_commodity_features(self.prices[name], self.day)
            commodity_feats.append(feat)

        # Cross-commodity correlations (6)
        cross_feats = compute_cross_commodity_features(
            self.prices, self.commodity_names, self.day
        )

        # Portfolio state (6)
        pv = self._portfolio_value()
        current_prices = {name: self.prices[name][self.day] for name in self.commodity_names}
        portfolio_feats = compute_portfolio_features(
            cash=self.cash,
            portfolio_value=pv,
            initial_cash=self.initial_cash,
            peak_value=self.peak_value,
            day=self.day,
            n_days=self.n_days,
            positions=self.positions,
            prices=current_prices,
            prev_positions=self.prev_positions,
        )

        return np.concatenate(commodity_feats + [cross_feats, portfolio_feats])

    def step(self, action):
        action = np.clip(action, -1, 1)
        pv_before = self._portfolio_value()

        # Save previous positions for turnover calculation
        self.prev_positions = {n: self.positions[n] for n in self.commodity_names}

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

                units = trade_value / price
                self.positions[name] += units
                self.cash -= trade_value + np.sign(trade_value) * spread * abs(units)
                self.cash -= commission
                total_cost += commission + abs(units) * spread

        # Advance day
        self.day += 1
        done = self.day >= self.n_days - 1

        # Portfolio value after price move
        pv_after = self._portfolio_value()
        self.portfolio_values.append(pv_after)
        self.peak_value = max(self.peak_value, pv_after)

        daily_return = (pv_after - pv_before) / pv_before if pv_before > 0 else 0.0
        self.daily_returns.append(daily_return)

        # --- Reward shaping ---
        # 1) Scaled daily return (primary signal)
        reward = daily_return * 100

        # 2) Drawdown penalty (moderate — don't overwhelm return signal)
        drawdown = (self.peak_value - pv_after) / self.peak_value
        reward -= drawdown * 3

        # 3) Mild turnover penalty (only penalize excessive churn)
        turnover = total_cost / pv_before if pv_before > 0 else 0
        reward -= turnover * 20

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
            returns = np.array(self.daily_returns)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            max_dd = 0
            peak = self.portfolio_values[0]
            for v in self.portfolio_values:
                peak = max(peak, v)
                dd = (peak - v) / peak
                max_dd = max(max_dd, dd)

            info["sharpe"] = round(sharpe, 3)
            info["max_drawdown"] = round(max_dd * 100, 2)
            info["final_value"] = round(pv_after, 2)

            # Terminal bonus: strong Sharpe signal
            reward += sharpe * 20 - max_dd * 5

        return self._get_obs(), reward, done, False, info
