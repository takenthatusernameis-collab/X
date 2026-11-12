"""Timing probe: one stress_segments call at realistic cost, to size the grid."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEGMENT_BARS = 60, 4, 400

REALISTIC = dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005)


def main() -> int:
    bars = bt.load_ticker("AMZN")[0]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=5)
    cfg0 = bt.BacktestConfig()
    cfg1 = bt.BacktestConfig(**REALISTIC)
    for name, cfg in [("zero", cfg0), ("realistic", cfg1)]:
        t0 = time.time()
        res = bt.stress_segments(
            signals_fn=bt.momentum_signals, bars=list(bars), signals=sig,
            param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
            segment_fn=seg_fn, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS, cfg=cfg,
        )
        dt = time.time() - t0
        print("{}: {:.2f}s verdict={} cand_medians={} null_medians={}".format(
            name, dt, res.overall_verdict,
            [round(s.baseline_median_log_return, 3) for s in res.scenarios],
            [round(s.noise_median_log_return, 3) for s in res.scenarios]))
    return 0


if __name__ == "__main__":
    sys.exit(main())