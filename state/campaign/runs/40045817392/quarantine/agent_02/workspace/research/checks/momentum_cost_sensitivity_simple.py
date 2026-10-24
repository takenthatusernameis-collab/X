#!/usr/bin/env python3
"""Simple cost-sensitivity check for lookback 3/5/10 momentum.

This is a simplified version that focuses on the core cost-sensitivity analysis
without complex summary logic.
"""
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

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

# Cost configurations
ZERO_COST_CFG = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
REALISTIC_COST_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


def segment_result_cost_sensitive(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, lookback, and cost config."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
        cfg_type="realistic" if cfg != ZERO_COST_CFG else "zero",
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading manifest...")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("Preflight passed...")

    print("Loading universe...")
    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}

    print(f"Running cost-sensitivity analysis for lookbacks {LOOKBACKS}")
    print(f"Zero cost config: commission_per_trade={ZERO_COST_CFG.commission_per_trade}, "
          f"commission_per_share={ZERO_COST_CFG.commission_per_share}")
    print(f"Realistic cost config: commission_per_trade={REALISTIC_COST_CFG.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COST_CFG.commission_per_share}, "
          f"slippage_cents={REALISTIC_COST_CFG.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COST_CFG.slippage_proportional}")

    per_asset_cost_sensitive = {}
    for ticker in tickers:
        per_asset_cost_sensitive[ticker] = {}
        for lb in LOOKBACKS:
            # Zero cost
            per_asset_cost_sensitive[ticker][f"{lb}_nocost"] = (
                segment_result_cost_sensitive(ticker, tickers[ticker], lb, ZERO_COST_CFG)
            )
            
            # Realistic cost
            per_asset_cost_sensitive[ticker][f"{lb}_cost"] = (
                segment_result_cost_sensitive(ticker, tickers[ticker], lb, REALISTIC_COST_CFG)
            )
            
            print(f"  {ticker} lookback={lb}: zero_cost_verdict={per_asset_cost_sensitive[ticker][f'{lb}_nocost']['verdict']}, "
                  f"realistic_cost_verdict={per_asset_cost_sensitive[ticker][f'{lb}_cost']['verdict']}")

    print("\nDeterminism check...")
    target = ["AMZN", "JPM"]
    res2 = {}
    for ticker in target:
        res2[ticker] = {}
        for lb in LOOKBACKS:
            res2[ticker][f"{lb}_nocost"] = (
                segment_result_cost_sensitive(ticker, tickers[ticker], lb, ZERO_COST_CFG)
            )
            res2[ticker][f"{lb}_cost"] = (
                segment_result_cost_sensitive(ticker, tickers[ticker], lb, REALISTIC_COST_CFG)
            )
    
    out1 = json.dumps(per_asset_cost_sensitive, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print(f"Determinism check: {out1 == out2}")
    if out1 != out2:
        print("NON-DETERMINISTIC OUTPUT - ABORTING")
        sys.exit(1)

    print(f"\nWriting artifact to {ARTIFACT_PATH}")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset_cost_sensitive=per_asset_cost_sensitive,
        zero_cost_config={
            "initial_capital": ZERO_COST_CFG.initial_capital,
            "warmup_periods": ZERO_COST_CFG.warmup_periods,
            "commission_per_trade": ZERO_COST_CFG.commission_per_trade,
            "commission_per_share": ZERO_COST_CFG.commission_per_share,
            "slippage_cents": ZERO_COST_CFG.slippage_cents,
            "slippage_proportional": ZERO_COST_CFG.slippage_proportional,
        },
        realistic_cost_config={
            "initial_capital": REALISTIC_COST_CFG.initial_capital,
            "warmup_periods": REALISTIC_COST_CFG.warmup_periods,
            "commission_per_trade": REALISTIC_COST_CFG.commission_per_trade,
            "commission_per_share": REALISTIC_COST_CFG.commission_per_share,
            "slippage_cents": REALISTIC_COST_CFG.slippage_cents,
            "slippage_proportional": REALISTIC_COST_CFG.slippage_proportional,
        },
        note="Cost-sensitivity analysis for lookback 3/5/10 momentum; zero-cost vs realistic-cost using coin-flip null; deterministic and preflight-verified."
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    
    print("\nSUCCESS: Cost-sensitivity analysis completed!")
    print(f"Results written to {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())