"""Synthetic bar generation for backtest-tooling validation.

Data is SYNTHETIC. It is intended only for validating the tooling
(engine, metrics, leakage checks). It is not real market data and must
not be used to claim a live strategy is profitable.

Design notes:
- daily bars: open/high/low/close/volume
- price follows a regime-switching geometric process with optional mean reversion
- each regime: annual drift, annual vol, mean-reversion strength/level, intraday range scale
- intraday range is derived from close (no future information in open/high/low)
- fully deterministic via seed
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

Bar = namedtuple(
    "Bar",
    ("date", "open", "high", "low", "close", "volume"),
)


@dataclass(frozen=True)
class Regime:
    """Parameter set for one market regime."""
    drift_annual: float = 0.03
    vol_annual: float = 0.20
    mean_reversion_speed: float = 0.0      # speed of mean reversion; >0 pulls price toward level
    mean_reversion_level: float = 100.0    # long-run price level (only used if speed > 0)
    p_transition: float = 0.01             # per-bar probability of leaving this regime
    intraday_range_scale: float = 0.02     # scale of intraday range relative to close


class BarSequence:
    """Container for a deterministic synthetic bar series."""

    def __init__(
        self,
        dates: np.ndarray,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> None:
        self.dates = dates
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.volumes = volumes
        assert len(self.dates) == len(self.opens) == len(self.highs) == len(self.lows)
        assert len(self.closes) == len(self.volumes)
        assert self.closes.size > 0

    @property
    def n_bars(self) -> int:
        return len(self.dates)

    def __iter__(self) -> Iterator[Bar]:
        for i in range(self.n_bars):
            yield Bar(
                int(self.dates[i]),
                float(self.opens[i]),
                float(self.highs[i]),
                float(self.lows[i]),
                float(self.closes[i]),
                float(self.volumes[i]),
            )

    def __len__(self) -> int:
        return self.n_bars

    def __getitem__(self, idx) -> Bar:
        i = int(idx)
        return Bar(self.dates[i], self.opens[i], self.highs[i], self.lows[i], self.closes[i], self.volumes[i])

    def closes_array(self) -> np.ndarray:
        return self.closes.copy()


def _generate_bars(
    n_bars: int,
    regimes: List[Regime],
    start_price: float,
    rng: np.random.Generator,
    trading_days_per_year: int = 252,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core synthetic data loop.

    Price follows regime-switching geometric dynamics with optional mean reversion.
    Returns (dates, opens, highs, lows, closes, volumes).
    """
    if not regimes:
        raise ValueError("at least one regime required")

    drift = [r.drift_annual / trading_days_per_year for r in regimes]
    vol = [r.vol_annual / np.sqrt(trading_days_per_year) for r in regimes]
    mr_speed = [r.mean_reversion_speed for r in regimes]
    mr_level = [r.mean_reversion_level for r in regimes]
    p_trans = [r.p_transition for r in regimes]
    range_scale = [r.intraday_range_scale for r in regimes]

    dates = np.arange(1, n_bars + 1, dtype=np.int64)
    opens = np.empty(n_bars, dtype=np.float64)
    highs = np.empty(n_bars, dtype=np.float64)
    lows = np.empty(n_bars, dtype=np.float64)
    closes = np.empty(n_bars, dtype=np.float64)
    volumes = np.empty(n_bars, dtype=np.float64)

    regime_idx = 0
    price = float(start_price)
    level = float(start_price)  # running log-level for mean-reversion target
    for t in range(n_bars):
        # regime dynamics: stay in regime with 1 - p_transition, else switch uniformly
        if rng.random() < p_trans[regime_idx]:
            regime_idx = rng.integers(0, len(regimes))

        r = regimes[regime_idx]
        # geometric Brownian motion step
        log_price = np.log(price)
        mu = drift[regime_idx]
        if mr_speed[regime_idx] > 0:
            # pull the long-run level toward the current regime mean-reversion level
            level = level + mr_speed[regime_idx] * (mr_level[regime_idx] - level) * (1.0 / trading_days_per_year)
            mu = mu + mr_speed[regime_idx] * (np.log(level) - log_price)

        shock = rng.standard_normal()
        logr = mu + vol[regime_idx] * shock
        price = price * np.exp(logr)

        # open = prior close; derive range from current close (no future info)
        open_p = start_price if t == 0 else closes[t - 1]
        r_up = rng.uniform(0.0, 1.0)
        r_down = rng.uniform(0.0, 1.0)
        intraday = range_scale[regime_idx]
        high_p = price * np.exp(abs(r_up - 0.5) * 2.0 * intraday)
        low_p = price * np.exp(-abs(r_down - 0.5) * 2.0 * intraday)
        high_p = max(high_p, open_p, price)
        low_p = min(low_p, open_p, price)

        # volume: lognormal baseline scaled by regime volatility
        volumes[t] = 1e6 * np.exp(rng.uniform(-0.5, 0.5)) * (1.0 + 2.0 * vol[regime_idx] / 0.20)

        opens[t] = open_p
        highs[t] = high_p
        lows[t] = low_p
        closes[t] = price

    return dates, opens, highs, lows, closes, volumes


def generate_bars(
    n_bars: int,
    regimes: Optional[List[Regime]] = None,
    p_transition: float = 0.01,
    start_price: float = 100.0,
    seed: Optional[int] = None,
    trading_days_per_year: int = 252,
) -> BarSequence:
    """Generate a deterministic synthetic daily bar series.

    Args:
        n_bars: number of daily bars to generate.
        regimes: list of regime parameter sets. Defaults to one
            low-vol mean-reverting regime and one high-vol regime.
        p_transition: per-bar probability of switching regimes.
        start_price: first close.
        seed: reproducible random seed.
        trading_days_per_year: used to annualize drift/vol.

    Returns:
        BarSequence with dates/opens/highs/lows/closes/volumes.
    """
    if regimes is None:
        regimes = [
            Regime(drift_annual=0.04, vol_annual=0.15, intraday_range_scale=0.015),
            Regime(drift_annual=-0.02, vol_annual=0.35, intraday_range_scale=0.030),
        ]

    rng = np.random.default_rng(seed)
    dates, opens, highs, lows, closes, volumes = _generate_bars(
        n_bars=n_bars,
        regimes=regimes,
        start_price=start_price,
        rng=rng,
        trading_days_per_year=trading_days_per_year,
    )
    return BarSequence(dates, opens, highs, lows, closes, volumes)
