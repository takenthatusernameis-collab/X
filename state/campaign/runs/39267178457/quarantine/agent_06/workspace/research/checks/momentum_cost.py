"""Cost-sensitivity check for lookback 3/5/10 momentum.

This test addresses the primary question: "Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe?"

Method (fixed a-priori, not tuned to OOS):
- Long the previous N-day return; hold 1 day; daily rebalance.
- Conservative transaction costs applied to both signal and matched null.
- Lookback horizons: 3, 5, 10 days.
- Same walk-forward parameters as existing momentum checks.

Conservative cost model (realistic but not live-trading):
- commission_per_trade: 0.01 (1 cent per trade)
- slippage_cents: 0.5 (50 cents fixed slippage per share)
- slippage_proportional: 0.001 (10 basis points)
- commission_per_share: 0.0 (covered by per-trade)

Evidence gate: The momentum edge must persist across all cost horizons to be considered robust.
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
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Conservative transaction cost model
COMMISSION_PER_TRADE = 0.01      # 1 cent per trade
SLIPPAGE_CENTS = 0.5             # 50 cents fixed slippage per share  
SLIPPAGE_PROPORTIONAL = 0.001     # 10 basis points proportional slippage
COMMISSION_PER_SHARE = 0.0       # Covered by per-trade commission

COST_CONFIG = bt.BacktestConfig(
    commission_per_trade=COMMISSION_PER_TRADE,
    commission_per_share=COMMISSION_PER_SHARE,
    slippage_cents=SLIPPAGE_CENTS,
    slippage_proportional=SLIPPAGE_PROPORTIONAL,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_results.json"

def momentum_signals_with_lookback(closes: np.ndarray, lookback: int) -> list:
    """Generate momentum signals with specific lookback period."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

def run_momentum_with_costs(asset: str, tickers: dict, lookback: int):
    """Run momentum with conservative costs for one asset and lookback."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    signals = momentum_signals_with_lookback(closes, lookback)
    
    if len(signals) != len(closes):
        raise ValueError(f"{asset} lookback={lookback}: signals {len(signals)} != bars {len(closes)}")
    
    res = bt.stress_segments(
        signals_fn=lambda c, **kw: momentum_signals_with_lookback(c, lookback),
        bars=list(bars),
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=("lookback", lookback),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=COST_CONFIG,
    )
    
    return {
        "lookback": lookback,
        "asset": asset,
        "segments": [s.name for s in res.scenarios],
        "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(res.null_dispersion, 3),
        "verdict": res.overall_verdict,
        "n_folds": res.n_folds,
        "cost_config": {
            "commission_per_trade": COST_CONFIG.commission_per_trade,
            "commission_per_share": COST_CONFIG.commission_per_share,
            "slippage_cents": COST_CONFIG.slippage_cents,
            "slippage_proportional": COST_CONFIG.slippage_proportional,
        }
    }

def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    
    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")
    
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)
    
    print("\n=== 3. Conservative cost model ===")
    print(f"  commission_per_trade: ${COST_CONFIG.commission_per_trade:.3f}")
    print(f"  commission_per_share: ${COST_CONFIG.commission_per_share:.3f}")
    print(f"  slippage_cents: ${COST_CONFIG.slippage_cents:.3f} per share")
    print(f"  slippage_proportional: {COST_CONFIG.slippage_proportional:.3%}")
    print(f"  Total cost impact per share: ${COST_CONFIG.commission_per_trade/100:.3f} + ${COST_CONFIG.slippage_cents/100:.3f} + proportional")
    
    print("\n=== 4. Momentum cost-sensitivity across lookback horizons ===")
    print(f"Lookback sweep: {LOOKBACKS} | Conservative transaction costs applied")
    print(f"Method: walk-forward per segment train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")
    print(f"Segments: {N_BLOCKS} contiguous volatility blocks, past-only classifier")
    print(f"Minimum segment bars: {MIN_SEGMENT_BARS}")
    
    # Load all tickers from the collected universe
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    
    # Run cost-sensitive momentum test for each lookback
    results = {}
    for lookback in LOOKBACKS:
        print(f"\n--- Lookback {lookback} days with conservative costs ---")
        
        # Universe-level analysis
        summary = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: momentum_signals_with_lookback(c, lookback),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lookback),),
            param_grid=[{"lookback": lookback}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=COST_CONFIG,
        )
        
        print(f"  {len(tickers)} assets | REGIME_STABLE={summary.verdict_counts.get('REGIME_STABLE', 0)} "
              f"CONSISTENT_WITH_NOISE={summary.verdict_counts.get('CONSISTENT_WITH_NOISE', 0)} "
              f"REGIME_DEPENDENT={summary.verdict_counts.get('REGIME_DEPENDENT', 0)} "
              f"REGIME_STABLE_LOSS={summary.verdict_counts.get('REGIME_STABLE_LOSS', 0)}")
        
        # Per-asset detailed results
        results[f"lookback_{lookback}"] = {
            "verdict_counts": summary.verdict_counts,
            "ticker_order": summary.ticker_order,
            "per_asset": {}
        }
        
        for ticker, r in summary.assets.items():
            results[f"lookback_{lookback}"]["per_asset"][ticker] = {
                "segments": [s.name for s in r.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in r.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in r.scenarios],
                "candidate_dispersion": round(r.candidate_dispersion, 3),
                "null_dispersion": round(r.null_dispersion, 3),
                "verdict": r.overall_verdict,
            }
            
            print(f"    {ticker:6s}: {r.overall_verdict:20s} medians {['%.3f' % m for m in [round(s.baseline_median_log_return, 3) for s in r.scenarios]]}")
    
    # Determinism check
    print("\n=== 5. Determinism verification ===")
    for lookback in LOOKBACKS:
        print(f"  Rerunning lookback {lookback} for determinism...")
        summary2 = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: momentum_signals_with_lookback(c, lookback),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lookback),),
            param_grid=[{"lookback": lookback}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
            cfg=COST_CONFIG,
        )
        
        # Compare with first run (simplified determinism check)
        verdict_counts_match = all(
            results[f"lookback_{lookback}"]["verdict_counts"][k] == v 
            for k, v in summary2.verdict_counts.items()
        )
        print(f"    lookback {lookback}: {'MATCH' if verdict_counts_match else 'MISMATCH'}")
    
    # Cost comparison to no-cost baseline
    print("\n=== 6. Cost impact analysis ===")
    print("  Note: Compare these results to the no-cost momentum baseline from the")
    print("        momentum check (lookback=5, all costs=0.0) in state/check_artifacts/momentum_results.json")
    print("  A robust edge should show similar verdict patterns across cost horizons.")
    
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_config={
            "commission_per_trade": COST_CONFIG.commission_per_trade,
            "commission_per_share": COST_CONFIG.commission_per_share,
            "slippage_cents": COST_CONFIG.slippage_cents,
            "slippage_proportional": COST_CONFIG.slippage_proportional,
        },
        **results,
    )
    
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print("\n=== 7. Artifact written ===")
    print(f"  Artifact written to: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Conservative cost-momentum check on the collected universe: exploratory simulation.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())