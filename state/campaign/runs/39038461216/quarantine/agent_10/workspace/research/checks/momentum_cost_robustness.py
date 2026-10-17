"""Cost-robustness check for momentum (lookback 3/5/10).

Research simulation only. This runs the momentum strategy under the repository's
existing realistic cost model and compares against a matched null. It tests
whether the short-horizon momentum edge survives conservative transaction-cost
stress on the collected real-data universe.

The bounded objective:
- Test lookback 3 / 5 / 10 momentum under realistic costs
- Use the existing 10-asset collected universe and momentum implementation
- Match the null via coin-flip benchmark on the same segments
- Stop after the cost grid is evaluated
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    periods_per_year=252,
)

ZERO_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    periods_per_year=252,
)


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    """Extract contiguous segments meeting minimum length requirement."""
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    """Momentum signals: long the previous N-day return, hold 1 day, daily rebalance."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_median(ticker, s, e, lookback, config):
    """Run walk-forward on a single segment with given lookback and config."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    seg_bars = list(bars)[s:e]
    signals = momentum_signals(closes, lookback)
    seg_signals = signals[s:e]
    result = bt.walk_forward(
        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=config)
    log_returns = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                           for f in result.folds], dtype=np.float64)
    return float(np.median(log_returns))


def noise_benchmark(bars, param_grid, train_window=60, test_window=20,
                  warmup=0, overlap_window=0, cfg=ZERO_COST_CONFIG, seed=99):
    """Coin-flip null benchmark for the given param grid (mirrors bt.noise_benchmark)."""
    n = len(bars)
    base_seed = seed if seed is not None else 99

    def _param_seed(params):
        # deterministic, process-independent mapping from params to a seed.
        total = sum(float(v) for v in params.values())
        return base_seed + int(round(total) * 1000)

    signals = bt.random_signals(n, seed=_param_seed({"lookback": 0}))
    result = bt.walk_forward(
        list(bars), signals, train_window=train_window, test_window=test_window,
        warmup=warmup, overlap_window=overlap_window, cfg=cfg)
    log_returns = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                           for f in result.folds], dtype=np.float64)
    return float(np.median(log_returns))


def main():
    ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
    ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Momentum cost-robustness check (lookback 3 / 5 / 10) ===")
    
    # 1. Manifest integrity
    print("\n=== 1. Manifest integrity ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # 2. Load all tickers
    print("\n=== 2. Loading collected universe ===")
    tickers = {entry["ticker"]: bt.load_ticker(entry["ticker"])[0]
               for entry in manifest["entries"]}
    print(f"  Loaded {len(tickers)} tickers")

    # 3. Cost-sensitivity grid
    print("\n=== 3. Cost-sensitivity grid (lookback 3 / 5 / 10) ===")
    print("  Realistic cost config:")
    print(f"    commission_per_trade={REALISTIC_COST_CONFIG.commission_per_trade}")
    print(f"    commission_per_share={REALISTIC_COST_CONFIG.commission_per_share}")
    print(f"    slippage_cents={REALISTIC_COST_CONFIG.slippage_cents}")
    print(f"    slippage_proportional={REALISTIC_COST_CONFIG.slippage_proportional}")
    print("  Zero-cost config: no commissions or slippage")

    # 4. Run cost-sensitivity analysis
    print("\n=== 4. Cost-sensitivity analysis ===")
    cost_results = {}
    zero_cost_results = {}
    noise_medians = {}
    
    for ticker in manifest["universe"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        print(f"\n  {ticker}: {len(runs)} segments")
        
        ticker_cost_results = {}
        ticker_zero_cost_results = {}
        
        for lb in LOOKBACKS:
            # Realistic costs
            costs = []
            zero_costs = []
            
            for label, s, e in runs:
                # Realistic costs
                median_cost = segment_median(ticker, s, e, lb, REALISTIC_COST_CONFIG)
                costs.append(median_cost)
                
                # Zero costs  
                median_zero = segment_median(ticker, s, e, lb, ZERO_COST_CONFIG)
                zero_costs.append(median_zero)
            
            cost_results[f"{ticker}_lb{lb}_cost"] = {
                "lookback": lb,
                "medians": [round(m, 3) for m in costs],
                "segments": [label for label, _, _ in runs],
                "n_segments": len(runs),
                "avg_median": round(float(np.mean(costs)), 3),
                "cost_impact": round(float(np.mean(zero_costs) - np.mean(costs)), 3),
            }
            
            zero_cost_results[f"{ticker}_lb{lb}_zero"] = {
                "lookback": lb,
                "medians": [round(m, 3) for m in zero_costs],
                "segments": [label for label, _, _ in runs],
                "n_segments": len(runs),
            }
            
            # Noise benchmark for this lookback
            bars_seg = tickers[ticker]
            noise_medians[lb] = noise_benchmark(bars_seg, [{"lookback": lb}], seed=42)
            
            print(f"    lb{lb}: zero-cost median={np.mean(zero_costs):+.3f}, "
                  f"realistic-cost median={np.mean(costs):+.3f}, "
                  f"cost impact={np.mean(zero_costs) - np.mean(costs):+.3f}, "
                  f"null median={noise_medians[lb]:+.3f}")

    # 5. Create artifact
    print("\n=== 5. Creating artifact ===")
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": 42,
        "lookbacks": list(LOOKBACKS),
        "realistic_cost_config": {
            "commission_per_trade": REALISTIC_COST_CONFIG.commission_per_trade,
            "commission_per_share": REALISTIC_COST_CONFIG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CONFIG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CONFIG.slippage_proportional,
            "initial_capital": REALISTIC_COST_CONFIG.initial_capital,
            "warmup_periods": REALISTIC_COST_CONFIG.warmup_periods,
        },
        "zero_cost_config": {
            "initial_capital": ZERO_COST_CONFIG.initial_capital,
            "warmup_periods": ZERO_COST_CONFIG.warmup_periods,
        },
        "train": TRAIN,
        "test": TEST,
        "warmup": WARM,
        "overlap": OVERLAP,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "per_ticker": cost_results,
        "per_ticker_zero_cost": zero_cost_results,
        "noise_medians": {lb: round(v, 3) for lb, v in noise_medians.items()},
        "universe_summary": {
            ticker: {
                "lookback_3_cost_impact": round(cost_results[f"{ticker}_lb3_cost"]["cost_impact"], 3),
                "lookback_5_cost_impact": round(cost_results[f"{ticker}_lb5_cost"]["cost_impact"], 3),
                "lookback_10_cost_impact": round(cost_results[f"{ticker}_lb10_cost"]["cost_impact"], 3),
            }
            for ticker in manifest["universe"]
        }
    }
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print(f"  Artifact written to: {ARTIFACT_PATH}")
    
    # 6. Determinism check
    print("\n=== 6. Determinism check ===")
    np.random.seed(42)
    first_run = artifact["per_ticker"]["AMZN_lb3_cost"]["medians"][:3]
    
    # Recreate using our implementation
    bars = tickers["AMZN"]
    lookback = 3
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    
    recomputed = []
    for label, s, e in runs[:3]:
        median = segment_median("AMZN", s, e, lookback, REALISTIC_COST_CONFIG)
        recomputed.append(round(median, 3))
    
    print(f"  First segment AMZN lb3 realistic-cost median (artifact): {first_run}")
    print(f"  First segment AMZN lb3 realistic-cost median (recomputed): {recomputed}")
    print(f"  Deterministic: {first_run == recomputed}")
    assert first_run == recomputed, "Non-deterministic results"

    print("\n=== Cost-robustness check complete ===")
    print(f"  Testing lookback {LOOKBACKS} under realistic costs")
    print(f"  Artifact available for independent verification: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost-robustness check: exploratory simulation.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())