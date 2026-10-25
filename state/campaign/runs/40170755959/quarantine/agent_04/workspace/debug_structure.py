#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, 'research/backtest')

# Load manifest
from research.data.preflight import load_manifest
manifest = load_manifest()

# Import the backtest module
import research.backtest as bt

# Get universe
DATA_SEGT_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
assert manifest["dataset_id"] == DATA_SEGT_ID, manifest["dataset_id"]

# Get tickers
tickers_univ = {}
for entry in manifest["entries"]:
    ticker = entry["ticker"]
    tickers_univ[ticker] = bt.load_ticker(ticker)[0]

# Define hold_effective_signals function
def hold_effective_signals(signals, hold):
    """Carry each signal for `hold` bars before the next rebalance."""
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < 5:  # LOOKBACK = 5
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - 5) % hold  # LOOKBACK = 5
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out

# Test the hold_effective_signals function
LOOKBACK = 5

print('Testing hold_effective_signals function:')
for h in [1, 2, 3, 5]:
    # Generate test data
    bars = bt.generate_bars(600, seed=42)
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=LOOKBACK)
    eff = []
    for i in range(len(sig)):
        if i < LOOKBACK:
            eff.append(sig[0].__class__(date=i + 1, weight=0.0))
        else:
            idx = i - (i - LOOKBACK) % h
            eff.append(sig[0].__class__(date=i + 1, weight=sig[idx].weight))
    
    print(f'hold={h}: sig[0]={sig[0]}, eff[0]={eff[0]}')
    print(f'hold={h}: sig[0].weight={sig[0].weight}, eff[0].weight={eff[0].weight}')

print()
print('Testing stress_segments_across_tickers:')

for hold in [1, 2, 3, 5]:
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, h=hold, **kw: hold_effective_signals(
            bt.momentum_signals(c, lookback=LOOKBACK), h),
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
    
    print(f'\nhold={hold}:')
    print(f'  summary type: {type(summary)}')
    print(f'  summary.ticker_order: {summary.ticker_order}')
    print(f'  summary.assets type: {type(summary.assets)}')
    
    for ticker in summary.ticker_order[:2]:
        print(f'  {ticker}:')
        print(f'    summary.assets[{ticker}] type: {type(summary.assets[ticker])}')
        print(f'    summary.assets[{ticker}].__dict__ keys: {list(summary.assets[ticker].__dict__.keys())}')
        print(f'    summary.assets[{ticker}].scenarios type: {type(summary.assets[ticker].scenarios)}')
        if summary.assets[ticker].scenarios:
            print(f'    summary.assets[{ticker}].scenarios[0] type: {type(summary.assets[ticker].scenarios[0])}')
            print(f'    summary.assets[{ticker}].scenarios[0].__dict__ keys: {list(summary.assets[ticker].scenarios[0].__dict__.keys())}')
            print(f'    summary.assets[{ticker}].scenarios[0].baseline_median_log_return: {summary.assets[ticker].scenarios[0].baseline_median_log_return}')
            print(f'    type of baseline_median_log_return: {type(summary.assets[ticker].scenarios[0].baseline_median_log_return)}')

# Now create the hold_sums structure like in the original script
hold_sums = {}
for hold in [1, 2, 3, 5]:
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, h=hold, **kw: hold_effective_signals(
            bt.momentum_signals(c, lookback=LOOKBACK), h),
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
    
    hold_sums[str(hold)] = dict(
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

print('\n\nhold_sums structure check:')
print('hold_sums[\"1\"][\"per_asset\"][\"AAPL\"][\"medians\"]:', hold_sums['1']['per_asset']['AAPL']['medians'])
print('type:', type(hold_sums['1']['per_asset']['AAPL']['medians']))

print('\nAfter copy and filter:')
test_prof = hold_sums['1']['per_asset']['AAPL'].copy()
test_prof = {k: v for k, v in test_prof.items() if k != 'segments'}
print(f'test_prof["medians"]:', test_prof['medians'])
print(f'type:', type(test_prof['medians']))

print('\nProfiles structure:')
ticker_order = hold_sums['1']['ticker_order']
profiles = {}
for t in ticker_order:
    asset_profiles = {}
    for h in [1, 2, 3, 5]:
        prof = hold_sums[str(h)]['per_asset'][t].copy()
        prof = {k: v for k, v in prof.items() if k != 'segments'}
        asset_profiles[str(h)] = prof
        print(f'{t} hold={h}: prof[\"medians\"] type = {type(prof["medians"])}, value = {prof["medians"]}')
    profiles[t] = asset_profiles

print('\nChecking profiles[t][str(h)][\"medians\"]:')
for t in ['AAPL', 'MSFT']:
    for h in [1, 2, 3, 5]:
        medians = profiles[t][str(h)]["medians"]
        print(f'{t} hold={h}: medians type = {type(medians)}, value = {medians}')