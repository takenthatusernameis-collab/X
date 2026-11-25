"""Diagnostic: clean cost-drag magnitude across lookbacks {3,5,10} for AMZN.

Reports (a) full-sample total return at each cost level (clean absolute drag,
no walk-forward denominator artifact) and (b) per-segment walk-forward medians
at full precision to confirm whether the regime-gate medians are genuinely
cost-insensitive or merely round to the same 3 decimals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400

COST = {
    "zero": dict(),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010),
}


def main() -> int:
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    for lb in (3, 5, 10):
        sig = bt.momentum_signals(closes, lookback=lb)
        print("=== lookback {} ===".format(lb))
        for name, cost in COST.items():
            cfg = bt.BacktestConfig(initial_capital=1e6, **cost)
            fr = bt.run_bars(list(bars), sig, cfg)
            total = (fr.equity_curve[-1] - 1e6) / 1e6
            res = bt.stress_segments(
                signals_fn=bt.momentum_signals, bars=bars, signals=sig,
                param_grid=[{"lookback": lb}], baseline=(("lookback", lb),),
                segment_fn=seg_fn, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, periods_per_year=252,
                min_segment_bars=MIN_SEG, cfg=cfg)
            meds = [round(s.baseline_median_log_return, 6) for s in res.scenarios]
            print("  {:12s} full_sample_total={:+.4f} seg_medians={} verdict={}".format(
                name, total, meds, res.overall_verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())