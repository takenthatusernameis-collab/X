"""Scratch smoke: time one ticker/lookback across the predeclared cost grid.

Representative subset only (AMZN, lookback 5) through stress_segments at the
three predeclared cost levels (zero / realistic / conservative). Confirms the
cost cfg flows through both the candidate sweep and the matched coin-flip null
before scaling to the full 10-asset x lookback{3,5,10} check.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

COST_LEVELS = {
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
    lookback = 5
    sig = bt.momentum_signals(closes, lookback=lookback)
    for name, cost in COST_LEVELS.items():
        cfg = bt.BacktestConfig(initial_capital=1e6, **cost)
        t0 = time.time()
        res = bt.stress_segments(
            signals_fn=bt.momentum_signals,
            bars=bars,
            signals=sig,
            param_grid=[{"lookback": lookback}],
            baseline=(("lookback", lookback),),
            segment_fn=seg_fn,
            train_window=TRAIN, test_window=TEST, warmup=WARM,
            overlap_window=OVERLAP, periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS, cfg=cfg,
        )
        dt = time.time() - t0
        meds = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
        nulls = [round(s.noise_median_log_return, 3) for s in res.scenarios]
        print("{}: segments={} medians={} nulls={} verdict={} ({:.1f}s)".format(
            name, [s.name for s in res.scenarios], meds, nulls,
            res.overall_verdict, dt))
    return 0


if __name__ == "__main__":
    sys.exit(main())