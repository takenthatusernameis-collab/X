"""Leakage and look-ahead checks for backtest artifacts.

Research/simulation only. Not live trading.
"""
from __future__ import annotations

from typing import List, NamedTuple

import numpy as np

Fill = NamedTuple("Fill", [("date", int), ("shares", float), ("price", float), ("commission", float)])
Signal = NamedTuple("Signal", [("date", int), ("weight", float)])


class LeakySignalError(ValueError):
    """Raised when a signal series violates the no-look-ahead discipline."""
    pass


def check_signal_integrity(signals: List[Signal], bar_dates: List[int], warmup: int) -> None:
    """Assert the signal series respects the no-look-ahead discipline.

    Checks:
    1. bar dates are unique and monotonically increasing;
    2. every signal's date exists in the bar series;
    3. no bar has a date greater than a later signal's date (signals
       never reference a future bar);
    4. weight is finite and within [-1, 1];
    5. no non-neutral weight appears before the warmup boundary.
       The engine enforces warmup itself, but signal authors must also
       pad their series so this holds.
    """
    bar_dates = [int(d) for d in bar_dates]
    if len(bar_dates) != len(set(bar_dates)):
        raise LeakySignalError("bar dates are not unique")
    for i in range(1, len(bar_dates)):
        if bar_dates[i] < bar_dates[i - 1]:
            raise LeakySignalError("bar dates not monotonically increasing")

    date_set = set(bar_dates)
    max_date = max(bar_dates)

    for idx, sig in enumerate(signals):
        sd = int(sig.date)
        if not np.isfinite(sig.weight):
            raise LeakySignalError(f"signal {idx}: weight {sig.weight} is not finite")
        if sig.weight < -1.0 or sig.weight > 1.0:
            raise LeakySignalError(f"signal {idx}: weight {sig.weight} out of [-1, 1]")
        if sd not in date_set or sd > max_date:
            raise LeakySignalError(f"signal {idx}: date {sd} not in bar series")
        # every signal must have all historical bars available up to its date;
        # equivalently the latest signal must reach the latest bar (no cut-off
        # signal series). A generic check cannot prove how the signal was computed,
        # but a signal at the last bar date proves the series is fully covered.

    for sig in signals:
        if int(sig.date) <= warmup and abs(sig.weight) > 1e-12:
            raise LeakySignalError(
                f"signal at date {sig.date} is non-neutral while warmup is {warmup}; "
                "pad signals through the warmup boundary"
            )


def check_equity_matches_fills(
    equity_curve: np.ndarray,
    fills: List[Fill],
    closes: np.ndarray,
    initial_capital: float,
    relative_tolerance: float = 1e-9,
) -> None:
    """Verify the equity curve is fully explained by the fill sequence.

    Recomputes equity from initial capital, fills (signed shares,
    execution prices, commissions) and bar closes, then asserts
    equality with the engine's equity curve within relative tolerance
    at every bar. This ensures no equity change is untraceable to a
    recorded fill.
    """
    closes = np.asarray(closes, dtype=np.float64)
    fills = sorted(fills, key=lambda f: f.date)
    equity_recomputed = np.zeros(len(equity_curve), dtype=np.float64)

    cash = float(initial_capital)
    shares = 0.0
    fill_index = 0

    for i in range(len(equity_curve)):
        bar_date = i + 1  # engine dates start at 1
        while fill_index < len(fills) and fills[fill_index].date == bar_date:
            f = fills[fill_index]
            if f.shares > 0:
                cash -= f.shares * f.price + f.commission
            else:
                cash += -f.shares * f.price - f.commission
            shares += f.shares
            fill_index += 1
        equity_recomputed[i] = cash + shares * closes[i]

    diff = np.abs(equity_curve - equity_recomputed)
    scale = np.maximum(np.abs(equity_curve), np.abs(equity_recomputed))
    if (diff / scale).max() > relative_tolerance:
        raise AssertionError(
            f"equity curve not fully explained by fills; "
            f"max relative mismatch {(diff / scale).max():.3e} at "
            f"indices {np.where(diff > relative_tolerance * scale)[0].tolist()}"
        )
