import sys
sys.path.insert(0, "/home/runner/work/X/X")
import numpy as np
import research.backtest as bt
from research.checks.momentum_hold_period_sensitivity import LOOKBACK

for ticker in ("AMZN", "AAPL", "JPM", "TSLA", "NVDA"):
    bars, _ = bt.load_ticker(ticker)
    closes = bars.closes_array()
    n = len(closes)
    sig = bt.momentum_signals(closes, lookback=LOOKBACK)
    w = np.array([s.weight for s in sig])
    frac_long = float(np.mean(w == 1.0))
    frac_short = float(np.mean(w == -1.0))
    frac_neutral = float(np.mean(w == 0.0))
    turns = int(np.sum(np.abs(np.diff(w)) > 0))

    # TRUE lookback-day log return sign at each bar
    true_ret = np.full(n, np.nan)
    for i in range(LOOKBACK, n):
        true_ret[i] = np.log(closes[i]) - np.log(closes[i - LOOKBACK])
    valid = ~np.isnan(true_ret)
    frac_true_pos = float(np.mean(true_ret[valid] > 0))
    # agreement between signal weight and true return sign
    agree = float(np.mean((w[valid] == 1.0) == (true_ret[valid] > 0)))

    # What a CORRECT return-based momentum signal would produce
    correct_w = np.where(true_ret > 0, 1.0, -1.0)
    correct_w[:LOOKBACK] = 0.0
    correct_turns = int(np.sum(np.abs(np.diff(correct_w)) > 0))
    correct_frac_long = float(np.mean(correct_w == 1.0))

    print(f"{ticker}: n={n}")
    print(f"  as-implemented momentum: frac_long={frac_long:.3f} frac_short={frac_short:.3f} "
          f"frac_neutral={frac_neutral:.3f} turns={turns}")
    print(f"  true 5d-return sign: frac_positive={frac_true_pos:.3f}  "
          f"signal/true agreement={agree:.3f}")
    print(f"  correct return-based momentum: frac_long={correct_frac_long:.3f} turns={correct_turns}")