"""Read-only timing probe for one asset / lookback / cost level.

Purpose: size the runtime of the bounded cost grid before running it. Uses the
same call shape as the planned check (stress_segments with the realistic cost
model), for one asset and one lookback.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

COST = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


def main() -> int:
    for ticker in ("AAPL", "AMZN"):
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)
        sig = bt.momentum_signals(closes, lookback=5)
        t0 = time.perf_counter()
        res = bt.stress_segments(
            signals_fn=bt.momentum_signals,
            bars=bars,
            signals=sig,
            param_grid=[{"lookback": 5}],
            baseline=(("lookback", 5),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=COST,
        )
        dt = time.perf_counter() - t0
        print("{}: {:.2f}s folds={} segments={} medians={}".format(
            ticker, dt, res.n_folds, [s.name for s in res.scenarios],
            [round(s.baseline_median_log_return, 3) for s in res.scenarios]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())