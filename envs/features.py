"""Enhanced feature engineering for commodity trading.

Computes per-commodity technical features and cross-commodity correlation
features. Designed to provide the transformer with rich market state
information beyond simple price-based indicators.

Feature groups:
  - Per-commodity (10 features × 4 commodities = 40):
    Returns, MA ratios, volatility, vol regime, RSI, Bollinger, momentum divergence
  - Cross-commodity (6 pairwise correlations)
  - Portfolio state (6 features: cash, return, drawdown, time, concentration, turnover)

Total observation dim: 52
"""

import numpy as np


N_FEATURES_PER_COMMODITY = 10
N_CROSS_FEATURES = 6  # pairwise correlations for 4 commodities
N_PORTFOLIO_FEATURES = 6
N_COMMODITIES = 4

STATE_DIM = N_COMMODITIES * N_FEATURES_PER_COMMODITY + N_CROSS_FEATURES + N_PORTFOLIO_FEATURES  # 52


def compute_commodity_features(prices: np.ndarray, idx: int) -> np.ndarray:
    """Compute technical features for a single commodity at time idx.

    Returns 10-dim feature vector:
        [ret_1d, ret_5d, ret_20d, price_vs_ma5, price_vs_ma20,
         realized_vol, vol_regime, rsi, bb_pos, momentum_divergence]
    """
    lookback = min(idx, 60)
    window = prices[max(0, idx - lookback + 1):idx + 1]
    current = prices[idx]

    # Returns
    ret_1d = (prices[idx] / prices[max(0, idx - 1)] - 1) if idx > 0 else 0.0
    ret_5d = (prices[idx] / prices[max(0, idx - 5)] - 1) if idx >= 5 else 0.0
    ret_20d = (prices[idx] / prices[max(0, idx - 20)] - 1) if idx >= 20 else 0.0

    # Moving averages
    ma5 = np.mean(prices[max(0, idx - 4):idx + 1])
    ma20 = np.mean(prices[max(0, idx - 19):idx + 1]) if idx >= 19 else ma5

    # Realized volatility (annualized, 20-day window)
    vol_window = prices[max(0, idx - 19):idx + 1]
    if len(vol_window) > 1:
        log_rets = np.diff(np.log(vol_window))
        realized_vol = np.std(log_rets) * np.sqrt(252)
    else:
        realized_vol = 0.0

    # Volatility regime: current vol vs longer-term vol distribution
    if idx >= 60:
        long_window = prices[max(0, idx - 59):idx + 1]
        long_rets = np.diff(np.log(long_window))
        rolling_vols = []
        for j in range(0, len(long_rets) - 19):
            rv = np.std(long_rets[j:j + 20]) * np.sqrt(252)
            rolling_vols.append(rv)
        if rolling_vols:
            vol_regime = (np.searchsorted(np.sort(rolling_vols), realized_vol)
                          / len(rolling_vols) - 0.5) * 2  # [-1, 1]
        else:
            vol_regime = 0.0
    else:
        vol_regime = 0.0

    # RSI (14-day)
    if idx >= 14:
        deltas = np.diff(prices[idx - 14:idx + 1])
        gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
        losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-10
        rsi = (100 - 100 / (1 + gains / (losses + 1e-10))) / 100 - 0.5
    else:
        rsi = 0.0

    # Bollinger band position
    bb_window = prices[max(0, idx - 19):idx + 1]
    if len(bb_window) > 1:
        std = np.std(bb_window)
        bb_pos = (current - np.mean(bb_window)) / (2 * std + 1e-10)
        bb_pos = np.clip(bb_pos, -2, 2) / 2
    else:
        bb_pos = 0.0

    # Momentum divergence: short-term vs long-term momentum
    # Positive = accelerating, Negative = decelerating
    short_mom = ret_5d
    long_mom = ret_20d / 4 if idx >= 20 else ret_5d  # normalize to similar scale
    momentum_div = np.clip(short_mom - long_mom, -0.2, 0.2) * 5  # scale to ~[-1, 1]

    return np.array([
        np.clip(ret_1d, -0.2, 0.2) * 5,        # scaled returns
        np.clip(ret_5d, -0.5, 0.5) * 2,
        np.clip(ret_20d, -1.0, 1.0),
        np.clip(current / ma5 - 1, -0.1, 0.1) * 10,   # MA ratios
        np.clip(current / ma20 - 1, -0.2, 0.2) * 5,
        np.clip(realized_vol, 0, 1.0) * 2 - 1,  # vol in [-1, 1]
        vol_regime,
        rsi,
        bb_pos,
        momentum_div,
    ], dtype=np.float32)


def compute_cross_commodity_features(all_prices: dict, commodity_names: list,
                                     idx: int, lookback: int = 20) -> np.ndarray:
    """Compute pairwise rolling correlations between commodities.

    Returns 6-dim vector (one correlation per pair).
    """
    n = len(commodity_names)
    returns = {}
    start = max(0, idx - lookback)

    for name in commodity_names:
        p = all_prices[name][start:idx + 1]
        if len(p) > 1:
            returns[name] = np.diff(np.log(p))
        else:
            returns[name] = np.array([0.0])

    correlations = []
    for i in range(n):
        for j in range(i + 1, n):
            r1 = returns[commodity_names[i]]
            r2 = returns[commodity_names[j]]
            min_len = min(len(r1), len(r2))
            if min_len > 2:
                corr = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            correlations.append(corr)

    return np.array(correlations, dtype=np.float32)


def compute_portfolio_features(cash: float, portfolio_value: float,
                               initial_cash: float, peak_value: float,
                               day: int, n_days: int,
                               positions: dict, prices: dict,
                               prev_positions: dict = None) -> np.ndarray:
    """Compute portfolio state features.

    Returns 6-dim vector:
        [cash_pct, total_return, drawdown, days_remaining, concentration, turnover]
    """
    pv = max(portfolio_value, 1.0)

    cash_pct = cash / pv
    total_return = np.clip(pv / initial_cash - 1, -1, 5)
    drawdown = np.clip((peak_value - pv) / peak_value, 0, 1) if peak_value > 0 else 0.0
    days_remaining = (n_days - day) / n_days

    # Portfolio concentration (Herfindahl index of absolute position weights)
    weights = []
    for name, units in positions.items():
        pos_value = abs(units * prices[name])
        weights.append(pos_value / pv if pv > 0 else 0)
    hhi = sum(w ** 2 for w in weights) if weights else 0
    # HHI range: [0, 1] where 1 = all in one asset, normalize to [-1, 1]
    concentration = hhi * 2 - 1

    # Turnover (how much positions changed)
    if prev_positions is not None:
        turnover = 0
        for name in positions:
            curr_val = abs(positions[name] * prices[name])
            prev_val = abs(prev_positions.get(name, 0) * prices[name])
            turnover += abs(curr_val - prev_val) / pv
        turnover = np.clip(turnover, 0, 1) * 2 - 1
    else:
        turnover = -1.0  # no turnover at start

    return np.array([
        cash_pct,
        total_return,
        drawdown,
        days_remaining,
        concentration,
        turnover,
    ], dtype=np.float32)
