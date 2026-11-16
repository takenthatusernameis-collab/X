"""Timing probe for the momentum cost-grid (scratch helper)."""
import sys, time
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400


def cfg_realistic():
    return bt.BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
                             slippage_cents=2.0, slippage_proportional=0.0005)


tickers = {}
for t in ["AMZN", "JPM"]:
    tickers[t] = bt.load_ticker(t)[0]

t0 = time.time()
summary = bt.stress_segments_across_tickers(
    tickers=tickers,
    signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=5),
    regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
    baseline=(("lookback", 5),),
    param_grid=[{"lookback": 5}],
    train_window=TRAIN, test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
    cfg=cfg_realistic(), min_segment_bars=MIN_SEG,
)
dt = time.time() - t0
print(f"2 assets 1 cell: {dt:.2f}s")
for t, r in summary.assets.items():
    print(t, r.overall_verdict,
          [round(s.baseline_median_log_return, 4) for s in r.scenarios],
          [round(s.noise_median_log_return, 4) for s in r.scenarios])