"""Volatility-regime filter on synthetic data (walk-forward IS/OOS).

A past-only realized-volatility regime filter: estimate recent
annualized volatility (standard deviation of log returns over a
lookback window) and hold the long position only in low-vol regimes.

Demonstrates:
- past-only indicator construction;
- warmup padding for the volatility window;
- a walk-forward split with out-of-sample testing;
- cost sensitivity (0 vs realistic costs);
- the leakage discipline checks.

Research/simulation only. The data is synthetic and must not be
interpreted as evidence about live markets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt


def volatility_regime_signals(closes: np.ndarray, vol_window: int = 20, threshold: float = 0.20):
    """Past-only volatility-regime signals.

    Annualized realized volatility is the standard deviation of log
    returns over `vol_window` bars, scaled to an annual basis. Weight is
    +1 (long) in low-vol regimes (realized vol below `threshold`) and 0
    (cash) otherwise. `threshold` is a fixed level, not fitted to the
    data.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    sqrt252 = np.sqrt(252.0)
    for i in range(int(vol_window) - 1, n):
        log_returns = np.log(closes[i - vol_window + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if realized_annual < threshold else 0.0)
    return signals


def main():
    np.random.seed(42)
    bars = bt.generate_bars(
        2500,
        regimes=[
            bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
            bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
        ],
        p_transition=0.008,
        start_price=100.0,
        seed=42,
    )

    vol_window, threshold = 20, 0.20
    signals = volatility_regime_signals(bars.closes_array(), vol_window, threshold)
    bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)

    warmup = vol_window
    cfg0 = bt.BacktestConfig(initial_capital=1e6, warmup_periods=warmup)
    cfg1 = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=warmup,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    print("=== Full-sample volatility-regime filter (regime-switching synthetic data) ===")
    for label, cfg in (("zero cost", cfg0), ("realistic costs", cfg1)):
        res = bt.run_bars(list(bars), signals, cfg)
        fill_prices = (
            np.array([f.price for f in res.trades], dtype=np.float64)
            if res.trades
            else np.array([], dtype=np.float64)
        )
        m = bt.compute_metrics(
            res.equity_curve,
            fills=res.trades,
            fill_prices=fill_prices,
            periods_per_year=252,
        )
        bt.check_equity_matches_fills(res.equity_curve, res.trades, bars.closes_array(), 1e6)
        print(f"[{label}] trades={m.n_trades} total={m.total_return:.2%} "
              f"sharpe={m.sharpe:.2f} max_dd={m.max_drawdown:.2%} "
              f"turnover={m.turnover_ratio:.1f}x")

    print()
    print("=== Walk-forward validation (train=12mo, test=4mo, warmup=20d) ===")
    result = bt.walk_forward(
        list(bars),
        signals,
        train_window=252,
        test_window=84,
        warmup=warmup,
        overlap_window=60,
        cfg=cfg0,
    )
    print(f"aggregates: folds={result.aggregate_metrics['n_folds']} "
          f"oos_periods={result.aggregate_metrics['total_oos_periods']} "
          f"mean_log_ret={result.aggregate_metrics['mean_log_total_return']:.3f} "
          f"median_log_ret={result.aggregate_metrics['median_log_total_return']:.3f}")

    for fold in result.folds:
        m = fold.metrics
        print(f"  fold {fold.fold_index + 1} [{fold.start_date}..{fold.end_date}]: "
              f"ret={m['total_return']:.2%} sharpe={m['sharpe']:.2f} "
              f"max_dd={m['max_drawdown']:.2%} trades={m['n_trades']}")

    fold_pos = sum(1 for f in result.folds if f.metrics["total_return"] > 0)
    print()
    print(f"positive OOS folds: {fold_pos}/{len(result.folds)}")
    print()
    print("NOTE: data is synthetic and intended only for tooling validation.")


if __name__ == "__main__":
    main()
