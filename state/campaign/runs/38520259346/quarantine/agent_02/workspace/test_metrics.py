#!/usr/bin/env python3
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

# Test the metrics structure
manifest = load_manifest()
entry = manifest['entries'][0]  # First ticker
bars, dates = bt.load_ticker(entry['ticker'])
closes = bars.closes_array()
sig = bt.momentum_signals(closes, lookback=5)

# Run a small walk_forward to see the metrics structure
result = bt.walk_forward(
    list(bars), sig, train_window=252, test_window=84,
    warmup=60, overlap_window=60, cfg=bt.BacktestConfig(initial_capital=1e6, warmup_periods=60)
)

print('Aggregate metrics keys:', result.aggregate_metrics.keys())
if result.folds:
    print('First fold metrics keys:', result.folds[0].metrics.keys())
    print('Sample fold metrics:', result.folds[0].metrics)