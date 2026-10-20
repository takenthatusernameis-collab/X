"""Smoke test for breakout_signals: one ticker, leakage check, one regime segment.

Reference path — the smallest representative run through the capability needed
for the activation. Runs before the full universe check.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

manifest = load_manifest()
assert manifest["dataset_id"] == "yf-ohlcv-universe-2009-to-2026-10-03"

ticker = "AAPL"
bars, dates = bt.load_ticker(ticker)
closes = bars.closes_array()
print(f"{ticker}: {bars.n_bars} bars, {len(closes)} closes")

for lookback in (5, 10, 20):
    sig = bt.breakout_signals(closes, lookback=lookback)
    assert len(sig) == len(closes), f"{ticker}: signals {len(sig)} != bars {len(closes)}"
    bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
    n_breakouts = sum(1 for s in sig if s.weight != 0.0)
    print(f"  lookback={lookback}: {n_breakouts} breakout bars of {len(closes)} ({100.0*n_breakouts/len(closes):.1f}%)")

# One representative segment run through the full gate machinery.
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)
sig = bt.breakout_signals(closes, lookback=20)
res = bt.stress_segments(
    signals_fn=bt.breakout_signals,
    bars=bars,
    signals=sig,
    param_grid=[{"lookback": 20}],
    baseline=(("lookback", 20),),
    segment_fn=seg_fn,
    train_window=252,
    test_window=84,
    warmup=60,
    overlap_window=60,
    periods_per_year=252,
    min_segment_bars=400,
)
print("  segments:", [s.name for s in res.scenarios])
for s in res.scenarios:
    print(f"    {s.name}: candidate {s.baseline_median_log_return:+.3f} vs null {s.noise_median_log_return:+.3f}")
print("  verdict:", res.overall_verdict)
print("  n_folds:", res.n_folds)
print("SMOKE OK")
sys.exit(0)
