"""Timing probe for one stress_segments call under a realistic cost cfg."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt

TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400


def corrected_momentum(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(int(lookback), n):
        r = float(__import__("numpy").log(closes[i]) - __import__("numpy").log(closes[i - int(lookback)]))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if r > 0 else -1.0)
    return out


def main():
    cfg = bt.BacktestConfig(initial_capital=1e6,
                            commission_per_trade=2.0, commission_per_share=0.003,
                            slippage_cents=2.0, slippage_proportional=0.0005)
    for t in ["AAPL", "AMZN"]:
        bars, _ = bt.load_ticker(t)
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg = bt.segment_fn_from_labels(labels)
        sig = corrected_momentum(closes, 5)
        t0 = time.time()
        res = bt.stress_segments(signals_fn=corrected_momentum, bars=bars, signals=sig,
                                 param_grid=[{"lookback": 5}], baseline=(("lookback", 5),),
                                 segment_fn=seg, train_window=TRAIN, test_window=TEST,
                                 warmup=WARM, overlap_window=OVERLAP, cfg=cfg,
                                 periods_per_year=252, min_segment_bars=MIN_SEG)
        dt = time.time() - t0
        print(f"{t}: {dt:.2f}s segments={[s.name for s in res.scenarios]} "
              f"medians={[round(s.baseline_median_log_return,3) for s in res.scenarios]} "
              f"null={[round(s.noise_median_log_return,3) for s in res.scenarios]} "
              f"verdict={res.overall_verdict}")


if __name__ == "__main__":
    main()