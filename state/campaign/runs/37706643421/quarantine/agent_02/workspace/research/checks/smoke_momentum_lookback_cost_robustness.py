"""Smoke test for momentum_lookback_cost_robustness: one ticker, leakage check, one regime segment, zero-cost and realistic cost.

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


def zero_cost_config():
    """Zero-cost config."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=60,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    )


def realistic_cost_config():
    """Realistic cost config."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=60,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )


for lookback in (3, 5, 10):
    for name, cfg in [("zero_cost", zero_cost_config()), ("realistic_cost", realistic_cost_config())]:
        sig = bt.momentum_signals(closes, lookback=lookback)
        assert len(sig) == len(closes), f"{ticker}: signals {len(sig)} != bars {len(closes)}"
        bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
        n_positive = sum(1 for s in sig if s.weight == 1.0)
        n_negative = sum(1 for s in sig if s.weight == -1.0)
        print(f"  {name}: lookback={lookback}: {n_positive} long, {n_negative} short of {len(closes)} ({100.0*n_positive/len(closes):.1f}% long)")

# One representative segment run through the full gate machinery for both cost configs.
closes = bars.closes_array()
labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)

for name, cfg in [("zero_cost", zero_cost_config()), ("realistic_cost", realistic_cost_config())]:
    for lookback in (3, 5, 10):
        sig = bt.momentum_signals(closes, lookback=lookback)
        res = bt.stress_segments(
            signals_fn=bt.momentum_signals,
            bars=bars,
            signals=sig,
            param_grid=[{"lookback": lookback}],
            baseline=(("lookback", lookback),),
            segment_fn=seg_fn,
            train_window=252,
            test_window=84,
            warmup=60,
            overlap_window=60,
            periods_per_year=252,
            min_segment_bars=400,
            cfg=cfg,
        )
        print("  {} lookback={}: segments {} -> {}".format(
            name, lookback, [s.name for s in res.scenarios], res.overall_verdict))
        for s in res.scenarios:
            print(f"    {s.name}: candidate {s.baseline_median_log_return:+.3f} vs null {s.noise_median_log_return:+.3f}")
        print("    verdict:", res.overall_verdict)
        print("    n_folds:", res.n_folds)

print("SMOKE OK")
sys.exit(0)