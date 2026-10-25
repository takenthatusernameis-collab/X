#!/usr/bin/env python3
import sys
sys.path.insert(0, 'research/backtest')

from research.backtest.regime_stability import stress_segments_across_tickers
from research.backtest import generate_bars, momentum_signals
import research.backtest as bt

# Create test data
bars = generate_bars(600, seed=42)
closes = bars.closes_array()

# Test the scenario directly
LOOKBACK = 5

# Create a simple test to reproduce the issue
for hold in [1, 2, 3, 5]:
    print(f'\n=== Testing hold={hold} ===')
    
    # Create a simple version of what hold_sums should look like
    summary = stress_segments_across_tickers(
        tickers={'AAPL': bars},
        signals_fn=lambda c, h=hold, **kw: momentum_signals(c, lookback=LOOKBACK),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=4, window=60),
        baseline=(("hold", float(hold)),),
        param_grid=[{"hold": hold}],
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
        periods_per_year=252,
        min_segment_bars=400,
    )
    
    # Check what summary.assets looks like
    print(f'summary.assets["AAPL"].__dict__ keys: {list(summary.assets["AAPL"].__dict__.keys())}')
    
    # Check what summary.assets["AAPL"].scenarios looks like
    print(f'summary.assets["AAPL"].scenarios type: {type(summary.assets["AAPL"].scenarios)}')
    print(f'summary.assets["AAPL"].scenarios[0] type: {type(summary.assets["AAPL"].scenarios[0])}')
    
    # Now let's see what happens when we try to create hold_sums entry
    hold_sums_entry = dict(
        verdict_counts=summary.verdict_counts,
        ticker_order=summary.ticker_order,
        per_asset=dict(
            (t, dict(
                segments=[s.name for s in r.scenarios],
                medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
                null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
                candidate_dispersion=round(r.candidate_dispersion, 3),
                null_dispersion=round(r.null_dispersion, 3),
                verdict=r.overall_verdict,
            ))
            for t, r in summary.assets.items()
        ),
    )
    
    print(f'hold_sums_entry["per_asset"]["AAPL"]["medians"] type: {type(hold_sums_entry["per_asset"]["AAPL"]["medians"])}')
    print(f'hold_sums_entry["per_asset"]["AAPL"]["medians"]: {hold_sums_entry["per_asset"]["AAPL"]["medians"]}')
    
    # Now test what happens when we copy and filter
    prof = hold_sums_entry["per_asset"]["AAPL"].copy()
    prof = {k: v for k, v in prof.items() if k != "segments"}
    
    print(f'prof["medians"] type: {type(prof["medians"])}')
    print(f'prof["medians"]: {prof["medians"]}')
    
    # Test the problematic code from line 445
    try:
        result = all(m > 0 for m in prof["medians"])
        print(f'Prof successfully processed: {result}')
    except TypeError as e:
        print(f'Error: {e}')
        print(f'prof["medians"] type: {type(prof["medians"])}')
        print(f'Is it a bool? {isinstance(prof["medians"], bool)}')