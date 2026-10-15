#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research.backtest as bt
from research.backtest import momentum_signals, stress_segments
from research.backtest import load_ticker
from research.data.preflight import load_manifest

manifest = load_manifest()
tickers_univ = {}
for entry in manifest['entries']:
    ticker = entry['ticker']
    tickers_univ[ticker] = bt.load_ticker(ticker)[0]

asset = manifest['universe'][0]
bars = tickers_univ[asset]
closes = bars.closes_array()

# Test momentum_signals
print('Testing momentum_signals...')
signals = momentum_signals(closes, lookback=5)
print(f'  Created {len(signals)} signals')

# Test stress_segments
print('Testing stress_segments...')
def signals_fn(c, **p):
    return momentum_signals(c, lookback=5)

labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
seg_fn = bt.segment_fn_from_labels(labels)

print('Calling stress_segments...')
try:
    cfg = bt.BacktestConfig(initial_capital=1e6, warmup_periods=60)
    baseline = ('lookback', 5)
    print(f'baseline type: {type(baseline)}, value: {baseline}')
    print(f'baseline as tuple: {tuple(baseline) if hasattr(baseline, "__iter__") else "not iterable"}')
    print(f'baseline wrapped: {(("lookback", 5),)}')
    res = stress_segments(
        signals_fn=signals_fn,
        bars=list(bars),
        signals=signals_fn(closes),
        param_grid=[{'lookback': 5}],
        baseline=(("lookback", 5),),
        segment_fn=seg_fn,
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
        periods_per_year=252,
        min_segment_bars=400,
        cfg=cfg,
    )
    print('Success!')
    print(f'  Verdict: {res.overall_verdict}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
