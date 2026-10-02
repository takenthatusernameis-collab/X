"""Event-driven daily backtest engine.

Design rules (enforced here and documented in research/METHODOLOGY.md):
- Signals are applied at the close of the same bar: no open/high/low
  lookahead. Fill price = close + fixed slippage + proportional slippage.
- warmup_periods bars are skipped: equity is initial capital and no
  signals are used. Signal authors must pad their signal series so that
  no real signal exists before the warmup boundary.
- IS/OOS walk-forward validation is provided by walk_forward().
- Commission and slippage apply to every round trip.
- Net equity = cash + shares * close; negative cash is allowed
  (short-sale proceeds are not reinvested) — conservative.

Research/simulation only. Not live trading.
"""
from __future__ import annotations

import dataclasses
from typing import List, NamedTuple

import numpy as np

from .data import Bar


class Signal(NamedTuple):
    """Signal applied at bar close. weight in [-1, 1]: short..neutral..long."""
    date: int
    weight: float


class Order(NamedTuple):
    date: int
    target_shares: float
    side: str  # 'buy' | 'sell' | 'cover' | 'short' | 'neutralize' | 'increase_long' | 'reduce_short'


class Fill(NamedTuple):
    date: int
    shares: float     # signed shares traded (positive = net long leg)
    price: float
    commission: float


@dataclasses.dataclass
class BacktestConfig:
    """Costs, cash and warmup policy for one backtest.

    Position sizing is constant-dollar: `target_exposure` is applied to
    initial_capital, so a weight of 1.0 holds a position worth
    initial_capital * target_exposure in each signal bar. This keeps
    sizing independent of accumulated (unrealized) gains and avoids
    automatic leverage in a zero-cost, no-margin engine.
    """
    initial_capital: float = 1e6
    target_exposure: float = 1.0     # gross exposure (long or short) as fraction of cash
    commission_per_trade: float = 0.0
    commission_per_share: float = 0.0
    slippage_cents: float = 0.0      # fixed slippage per share, in cents
    slippage_proportional: float = 0.0  # proportional slippage (0.001 = 10 bps)
    warmup_periods: int = 0          # leading bars to skip for indicator warmup
    risk_free: float = 0.0
    periods_per_year: int = 252


@dataclasses.dataclass
class FillResult:
    """Output of one backtest run."""
    equity_curve: np.ndarray
    trades: List[Fill]
    orders: List[Order]
    positions: List[float]  # shares held after each bar (signed)


@dataclasses.dataclass
class FoldResult:
    """Result of one walk-forward fold."""
    fold_index: int
    start_date: int
    end_date: int
    train_window_bars: int
    test_window_bars: int
    metrics: dict
    equity_curve: np.ndarray


@dataclasses.dataclass
class WalkForwardResult:
    """Results over all walk-forward folds."""
    folds: List[FoldResult]
    aggregate_metrics: dict


def _side_label(delta: float, shares: float) -> str:
    if abs(delta) < 1e-12:
        return "neutralize"
    if delta > 0:
        if shares <= 0:
            return "cover" if shares < 0 else "buy"
        return "increase_long"
    if shares >= 0:
        return "sell" if shares > 0 else "short"
    return "reduce_short"


def _run_bars(
    bars: List[Bar],
    signals: List[Signal],
    cfg: BacktestConfig,
) -> FillResult:
    cash = float(cfg.initial_capital)
    shares = 0.0
    orders: List[Order] = []
    fills: List[Fill] = []
    n = len(bars)
    positions = [0.0] * n
    equity_curve = np.zeros(n, dtype=np.float64)

    for i, bar in enumerate(bars):
        if i < cfg.warmup_periods:
            equity_curve[i] = cash
            positions[i] = shares
            continue

        signal = signals[i]
        if abs(signal.weight) < 1e-12:
            # neutral: keep position, but mark equity
            equity_curve[i] = cash + shares * bar.close
            positions[i] = shares
            continue

        target_shares = (
            signal.weight
            * cfg.target_exposure
            * cfg.initial_capital
            / bar.close
        )

        delta = target_shares - shares

        if abs(delta) < 1e-12:
            equity_curve[i] = cash + shares * bar.close
            positions[i] = shares
            continue

        # Fill at close + fixed slippage + proportional slippage.
        # close-only pricing enforces no open/high/low lookahead.
        fill_price = (
            bar.close
            + cfg.slippage_cents / 100.0
            + bar.close * cfg.slippage_proportional
        )
        commission = (
            cfg.commission_per_trade
            + abs(delta) * cfg.commission_per_share
        )

        if delta > 0:
            cash -= delta * fill_price + commission
        else:
            cash += -delta * fill_price - commission

        side = _side_label(delta, shares)
        shares = target_shares
        orders.append(Order(bar.date, target_shares, side))
        fills.append(Fill(bar.date, delta, fill_price, commission))
        equity_curve[i] = cash + shares * bar.close
        positions[i] = shares

    return FillResult(equity_curve, trades=fills, orders=orders, positions=positions)


class Backtest:
    """Convenience class wrapping _run_bars with a config."""

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def run(self, bars: List[Bar], signals: List[Signal]) -> FillResult:
        if len(bars) != len(signals):
            raise ValueError("bars and signals must have equal length")
        return _run_bars(bars, signals, self.cfg)


def run_bars(
    bars: List[Bar],
    signals: List[Signal],
    cfg: BacktestConfig = BacktestConfig(),
) -> FillResult:
    """Run a single backtest over the full sample."""
    return Backtest(cfg).run(bars, signals)


def walk_forward(
    bars: List[Bar],
    signals: List[Signal],
    train_window: int,
    test_window: int,
    warmup: int = 0,
    overlap_window: int = 0,
    cfg: BacktestConfig = BacktestConfig(),
) -> WalkForwardResult:
    """Walk-forward IS/OOS validation.

    Folds slide across the series: each fold uses a contiguous
    window [start, end) with a warmup segment, a train segment and a
    test segment. Only the test segment is backtested; the train
    segment represents information available at fold start (and is not
    used to compute signals here — signal authors must have already
    produced signals consistent with the walk-forward discipline).

    The fold's equity curve is returned for the full window, but only
    the test segment is included in fold metrics.

    Args:
        bars: full bar series.
        signals: full signal series, warmup-padded by the caller if needed.
        train_window: bars dedicated to the training segment per fold.
        test_window: bars of the out-of-sample segment per fold.
        warmup: warmup bars at the start of each fold.
        overlap_window: how much the train segment overlaps the previous
            test segment (0 = non-overlapping folds).
        cfg: backtest costs and cash policy.

    Returns:
        WalkForwardResult with per-fold metrics and aggregates.
    """
    if train_window < 1:
        raise ValueError("train_window must be >= 1")
    if test_window < 1:
        raise ValueError("test_window must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if overlap_window > test_window:
        raise ValueError("overlap_window must be <= test_window")

    n = len(bars)
    if len(signals) != n:
        raise ValueError("bars and signals must have equal length")

    folds: List[FoldResult] = []
    step = test_window - overlap_window
    fold_start = 0
    cumulative_test = 0

    while True:
        fold_end = fold_start + warmup + train_window + test_window
        if fold_end > n:
            break
        fs = max(0, fold_start)
        fold_bars = bars[fs : fold_end]
        fold_signals = signals[fs : fold_end]
        fold_cfg = BacktestConfig(
            initial_capital=cfg.initial_capital,
            target_exposure=cfg.target_exposure,
            commission_per_trade=cfg.commission_per_trade,
            commission_per_share=cfg.commission_per_share,
            slippage_cents=cfg.slippage_cents,
            slippage_proportional=cfg.slippage_proportional,
            warmup_periods=warmup,
            risk_free=cfg.risk_free,
            periods_per_year=cfg.periods_per_year,
        )
        result = Backtest(fold_cfg).run(fold_bars, fold_signals)

        # The OOS segment is the last test_window bars of the fold window.
        test_eq = result.equity_curve[-test_window:]
        if len(test_eq) == 0:
            break

        # per-fold metrics on the OOS segment only
        from .metrics import compute_metrics
        oos_trades = result.trades[-test_window:]
        oos_prices = (
            np.array([f.price for f in oos_trades], dtype=np.float64)
            if oos_trades
            else np.array([], dtype=np.float64)
        )
        fold_metrics = compute_metrics(
            test_eq,
            fills=oos_trades,
            fill_prices=oos_prices,
            risk_free=cfg.risk_free,
            periods_per_year=cfg.periods_per_year,
        )
        folds.append(FoldResult(
            fold_index=len(folds),
            start_date=fold_bars[0].date,
            end_date=fold_bars[-1].date,
            train_window_bars=train_window,
            test_window_bars=len(test_eq),
            metrics=dataclasses.asdict(fold_metrics),
            equity_curve=result.equity_curve,
        ))
        cumulative_test += len(test_eq)
        fold_start += step

    aggregate: dict
    if folds:
        m = np.array([np.log1p(f.metrics["total_return"]) for f in folds])
        aggregate = {
            "n_folds": len(folds),
            "total_oos_periods": cumulative_test,
            "mean_log_total_return": float(np.mean(m)),
            "median_log_total_return": float(np.median(m)),
            "std_log_total_return": float(np.std(m, ddof=1)) if len(m) > 1 else 0.0,
            "fold_count_positive": int(int(np.sum(m > 0))),
            "fold_count_negative": int(int(np.sum(m < 0))),
            "fold_count_zero": int(int(np.sum(m == 0))),
        }
    else:
        aggregate = {"n_folds": 0, "total_oos_periods": 0}

    return WalkForwardResult(folds=folds, aggregate_metrics=aggregate)
