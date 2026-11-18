"""Smoke test for momentum cost sensitivity: one ticker, cost-grid spot check.

Runs the smallest representative slice through the cost machinery before the
full universe check. Reference path only.
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

for lookback in (3, 5, 10):
    for label, comm in (("zero", 0.0), ("comm_500", 500.0)):
        sig = bt.momentum_signals(closes, lookback=lookback)
        assert len(sig) == len(closes), f"{ticker}: signals {len(sig)} != bars {len(closes)}"
        bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
        labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
        seg_fn = bt.segment_fn_from_labels(labels)
        cfg = bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0,
                                commission_per_trade=comm, slippage_cents=0.0)
        res = bt.stress_segments(
            signals_fn=bt.momentum_signals, bars=bars, signals=sig,
            param_grid=[{"lookback": lookback}], baseline=(("lookback", lookback),),
            segment_fn=seg_fn, train_window=252, test_window=84,
            warmup=60, overlap_window=60, cfg=cfg, periods_per_year=252,
            min_segment_bars=400,
        )
        meds = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
        nulls = [round(s.noise_median_log_return, 3) for s in res.scenarios]
        print(f"  lookback={lookback} cost={label:8s}: medians={meds} nulls={nulls} "
              f"verdict={res.overall_verdict}")

print("SMOKE OK")
sys.exit(0)