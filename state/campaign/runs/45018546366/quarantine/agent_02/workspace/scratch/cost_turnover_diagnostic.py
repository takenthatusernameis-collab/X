"""Diagnostic: verify the realistic cost model actually bites on momentum.

Reports, for one asset (AMZN) at lookback 5 over the full sample, the number of
signal flips, number of fills, gross vs net full-sample total return at cost
scales 0/1/2, and the realized cost drag. This confirms the cost-sensitivity
result is not a harness artifact (e.g. costs silently not applied).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

BASE = dict(commission_per_trade=2.0, commission_per_share=0.003,
            slippage_cents=2.0, slippage_proportional=0.0005)


def cfg(scale):
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=BASE["commission_per_trade"] * scale,
        commission_per_share=BASE["commission_per_share"] * scale,
        slippage_cents=BASE["slippage_cents"] * scale,
        slippage_proportional=BASE["slippage_proportional"] * scale,
    )


def main():
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=5)
    weights = np.array([s.weight for s in sig])
    nonzero = weights[weights != 0.0]
    flips = int(np.sum(nonzero[1:] != nonzero[:-1]))
    total_bars = int(np.sum(weights != 0.0))
    print(f"AMZN lookback 5: bars={len(closes)} nonzero_signal_bars={total_bars} "
          f"signal_flips={flips} flips_per_bar={flips/total_bars:.4f}")
    for scale in (0.0, 1.0, 2.0):
        res = bt.run_bars(list(bars), sig, cfg(scale))
        tr = (res.equity_curve[-1] - 1e6) / 1e6
        n_fills = len(res.trades)
        gross_notional = float(np.sum([abs(f.shares) * f.price for f in res.trades]))
        commission = float(np.sum([f.commission for f in res.trades]))
        # slippage cost = sum over fills of |shares| * close * slippage_proportional
        print(f"  scale {scale}: total_return={tr:+.4f} fills={n_fills} "
              f"gross_traded_notional={gross_notional:,.0f} "
              f"commission_paid={commission:,.2f} "
              f"turnover_x_capital={gross_notional/1e6:.2f}")


if __name__ == "__main__":
    main()