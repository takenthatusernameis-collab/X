"""Diagnostic: locate non-determinism in regime-filtered momentum per-asset gate."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research.backtest as bt
from research.data.preflight import load_manifest

TARGET = ["AMZN", "JPM"]
VARIANTS = [("base", None), ("turbulent_only", "turbulent"), ("calm_only", "calm")]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACK = 5

manifest = load_manifest()
all_tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}

# Run the universe sweep once BEFORE the first gate run (as the check does)
summary0 = bt.stress_segments_across_tickers(
    tickers=all_tickers,
    signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=LOOKBACK),
    regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
    baseline=(("lookback", LOOKBACK),),
    param_grid=[{"lookback": LOOKBACK}],
    train_window=TRAIN, test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
    periods_per_year=252, min_segment_bars=MIN_SEGMENT_BARS,
)
print("universe sweep verdict_counts (sanity):", summary0.verdict_counts)

tickers = {a: all_tickers[a] for a in TARGET}


def run_asset(asset, keep=None):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    if keep is None:
        signals_fn = lambda c, **p: bt.momentum_signals(c, lookback=LOOKBACK)
    else:
        signals_fn = lambda c, **p: bt.regime_filtered_momentum_signals(c, keep=keep)
    signals = signals_fn(closes)
    if len(signals) != len(closes):
        raise ValueError("signal length mismatch")
    res = bt.stress_segments(
        signals_fn=signals_fn, bars=list(bars), signals=signals,
        param_grid=[{"lookback": LOOKBACK}], baseline=(("lookback", LOOKBACK),),
        segment_fn=seg_fn, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    names = [s.name for s in res.scenarios]
    medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
    return dict(
        segments=names,
        medians=medians,
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
    )


results1 = {}
for var_name, keep in VARIANTS:
    results1[var_name] = {a: run_asset(a, keep) for a in TARGET}

results2 = {}
for var_name, keep in VARIANTS:
    results2[var_name] = {a: run_asset(a, keep) for a in TARGET}

print("=== RUN 1 ===")
print(json.dumps(results1, sort_keys=True, indent=2))
print("=== RUN 2 ===")
print(json.dumps(results2, sort_keys=True, indent=2))
print("=== MISMATCH DETECTION ===")
mismatch_found = False
for var_name in [v[0] for v in VARIANTS]:
    for asset in TARGET:
        if json.dumps(results1[var_name][asset], sort_keys=True) != json.dumps(results2[var_name][asset], sort_keys=True):
            mismatch_found = True
            print(f"MISMATCH: {var_name} / {asset}")
            print("run1:", json.dumps(results1[var_name][asset], sort_keys=True))
            print("run2:", json.dumps(results2[var_name][asset], sort_keys=True))
print("determinism:", json.dumps(results1, sort_keys=True) == json.dumps(results2, sort_keys=True))
if mismatch_found:
    sys.exit(1)
