"""Smoke test for momentum cost-robustness check.

Quick validation that the check runs and produces sensible output on a small scale.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import research.backtest as bt


def test_smoke():
    print("=== Smoke test for momentum cost-robustness check ===")

    # Test momentum_signals function directly
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
    signals = bt.momentum_signals(closes, lookback=3)
    print(f"  momentum_signals on {len(closes)} bars: {len(signals)} signals")
    print(f"  signal weights: {[s.weight for s in signals[-5:]]}")

    # Test with realistic costs
    cfg = bt.BacktestConfig(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
        initial_capital=1e6,
        target_exposure=1.0,
    )
    print(f"  Realistic costs config: commission={cfg.commission_per_trade} + {cfg.commission_per_share}/share, "
          f"fixed slippage={cfg.slippage_cents}¢, proportional slippage={cfg.slippage_proportional:.4f}")

    # Test BacktestConfig creation
    bars = [bt.Bar(i, 100.0, 101.0, 99.0, 100.5, 1000.0) for i in range(1, 21)]
    sig = [bt.Signal(1, 1.0), bt.Signal(2, -1.0)] + [bt.Signal(i, 0.0) for i in range(3, 21)]
    try:
        result = bt.run_bars(bars, sig, cfg)
        print(f"  Backtest runs successfully: equity curve length {len(result.equity_curve)}")
    except Exception as e:
        print(f"  Backtest error: {e}")
        return 1

    print("  ✓ Smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(test_smoke())