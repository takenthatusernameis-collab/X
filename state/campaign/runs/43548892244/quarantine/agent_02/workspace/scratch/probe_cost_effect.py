"""Read-only probe: does the cost model bite, and what is the turnover?

Also confirms the momentum signal composition across the whole collected
universe (all closes above 1.0 -> mean log-price > 0 -> weight always +1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEGMENT_BARS = 60, 4, 400

LEVELS = {
    "zero": dict(),
    "1x": dict(commission_per_trade=2.0, commission_per_share=0.003,
               slippage_cents=2.0, slippage_proportional=0.0005),
    "5x": dict(commission_per_trade=10.0, commission_per_share=0.015,
               slippage_cents=10.0, slippage_proportional=0.0025),
}


def seg_result(bars, sig, cfg):
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals, bars=bars, signals=sig,
        param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
        segment_fn=seg_fn, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS, cfg=cfg,
    )
    return [round(s.baseline_median_log_return, 6) for s in res.scenarios]


def main() -> int:
    manifest = load_manifest()
    print("=== signal composition over the collected universe (lookback 5) ===")
    for entry in manifest["entries"]:
        t = entry["ticker"]
        bars, _ = bt.load_ticker(t)
        c = bars.closes_array()
        w = np.array([s.weight for s in bt.momentum_signals(c, 5)])
        print("  {:6s} n={} min_close={:.4f} long={:.4f} short={:.4f} neutral={}".format(
            t, len(c), float(c.min()), float(np.mean(w > 0)),
            float(np.mean(w < 0)), int(np.sum(w == 0))))

    print("\n=== cost effect on segment medians (AMZN, lookback 5) ===")
    bars, _ = bt.load_ticker("AMZN")
    sig = bt.momentum_signals(bars.closes_array(), 5)
    for name, kw in LEVELS.items():
        cfg = bt.BacktestConfig(**kw)
        print("  {:4s}: {}".format(name, seg_result(bars, sig, cfg)))

    print("\n=== full-sample total return / turnover (AMZN, lookback 5) ===")
    for name, kw in LEVELS.items():
        cfg = bt.BacktestConfig(**kw)
        res = bt.run_bars(list(bars), sig, cfg)
        closes = bars.closes_array()
        total = (res.equity_curve[-1] - 1e6) / 1e6
        notional = sum(abs(f.shares) * f.price for f in res.trades)
        costs = sum(f.commission for f in res.trades)
        n = len(closes)
        print("  {:4s}: total_return={:+.4f} mean_turnover_per_bar={:.4%} "
              "total_cost={:.0f} ({:.4%} of capital over {} bars)".format(
                  name, total, notional / 1e6 / n, costs, costs / 1e6, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())