#!/usr/bin/env python3
"""Cost-robust momentum check for lookback 3/5/10.

This script executes the bounded cost-sensitivity check for lookback 3/5/10
momentum under the repository's existing realistic cost model and matched null.

Objective: Assess whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe.

Scope: Use the existing 10-asset collected universe and existing momentum
implementation/check framework.

Out of scope: Do not tune signal rules after seeing cost results, expand the
universe, or introduce leverage.
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

# Realistic cost model from ma_crossover_real_data.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=WARM,
)

# Zero-cost model for matched null comparison
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=WARM,
)

ARTIFACT_DIR = Path.cwd() / "state" / "campaign" / "runs" / "40170755959" / "agents"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_robustness_check.json"

def load_manifest_and_verify():
    """Load manifest and verify dataset integrity."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, f"manifest id mismatch: {manifest['dataset_id']}"
    return manifest

def get_ticker_universe(manifest):
    """Get all tickers from the manifest (10-asset universe)."""
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    return tickers, list(tickers.keys())

def momentum_signals(closes, lookback):
    """Past-only short-horizon momentum signal (from regime_stability.py)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out

def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker and one lookback with given costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = momentum_signals(closes, lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=momentum_signals,
        bars=list(bars),
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
        cfg=cfg,
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
    )

def lookback_verdict(sr):
    """Verdict for one lookback's scenario result."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"

def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset.
    
    ROBUST      : uniformly positive REGIME_STABLE edges at >= 2 lookbacks.
    SENSITIVE   : a positive REGIME_STABLE edge at exactly one lookback only.
    NO_EDGE     : no lookback shows a positive REGIME_STABLE edge.
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Cost-robust momentum check (lookback 3 / 5 / 10) ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Realistic costs: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")

    # 1. Manifest integrity and universe loading
    print("\n=== 1. Manifest integrity ===")
    manifest = load_manifest_and_verify()
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    # 2. Data preflight
    print("\n=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting cost-robustness check.")
        sys.exit(1)
    print("  [PASS] preflight checks completed")

    # 3. Cost-robust momentum lookback sweep
    print("\n=== 3. Cost-robust momentum lookback sweep (3 / 5 / 10) ===")
    tickers, ticker_order = get_ticker_universe(manifest)

    # Results indexed [lookback][ticker][segment index].
    realistic_results = {lb: {t: segment_result(t, tickers[t], lb, REALISTIC_COSTS) 
                             for t in ticker_order} for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        profile = {str(lb): realistic_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts[v] += 1
        r5 = profile["5"]
        lookback_5_counts[r5["verdict"]] += 1
        # Store for artifact
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        per_asset[ticker] = profile

    print(f"  lookback 5 (reference) verdict counts across universe: {lookback_5_counts}")
    print(f"  cost-robustness verdict counts: "
          f"ROBUST={robustness_counts['ROBUST']} (>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} (edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}")

    print("  Per-asset cost-robustness profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
              "edge_counts={}".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"]))

    # 4. Synthetic perturbation sweep (for cost-aware null comparison)
    print("\n=== 4. Synthetic perturbation sweep (lookback 3 / 5 / 10) with realistic costs ===")
    bars = bt.generate_bars(600, seed=SEED)
    grid = [{"lookback": 3}, {"lookback": 5}, {"lookback": 10}]
    sweep_results = []
    null_results = []
    for params in grid:
        signals = momentum_signals(bars.closes_array(), params["lookback"])
        # With realistic costs
        sweep_result = bt.walk_forward(
            list(bars), signals, train_window=60, test_window=20,
            warmup=10, overlap_window=10, cfg=REALISTIC_COSTS)
        log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                        for f in sweep_result.folds], dtype=np.float64)
        sweep_results.append(round(float(np.median(log)), 3))
        
        # With zero costs (matched null)
        signals_zero = momentum_signals(bars.closes_array(), params["lookback"])
        null_result = bt.walk_forward(
            list(bars), signals_zero, train_window=60, test_window=20,
            warmup=10, overlap_window=10, cfg=ZERO_COSTS)
        null_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                           for f in null_result.folds], dtype=np.float64)
        null_results.append(round(float(np.median(null_log)), 3))

    print(f"  param sets: {grid}")
    print(f"  realistic cost medians: {sweep_results}")
    print(f"  zero-cost null medians: {null_results}")
    print(f"  realistic vs null: {[round(s - n, 3) for s, n in zip(sweep_results, null_results)]}")

    # 5. Determinism check
    print("\n=== 5. Determinism check ===")
    # Use only AMZN and JPM as reference subset for determinism
    target = ["AMZN", "JPM"]
    realistic_target = {lb: {t: segment_result(t, tickers[t], lb, REALISTIC_COSTS)
                             for t in target} for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: realistic_results[lb][t] for t in target}
                      for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(realistic_target, sort_keys=True)
    print(f"  determinism: realistic_results == target_subset: {out1 == out2}")
    assert out1 == out2, "non-deterministic cost-robustness output"

    # 6. Artifact creation
    print("\n=== 6. Artifact creation ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_costs=dict(
            commission_per_trade=REALISTIC_COSTS.commission_per_trade,
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t]["lookback_medians" if hasattr(per_asset[t], "lookback_medians") else str(lb)]["medians"] for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t]["lookback_null_medians" if hasattr(per_asset[t], "lookback_null_medians") else str(lb)]["null_medians"] for lb in LOOKBACKS},
                    verdicts_5=per_asset[t]["5"]["verdict"],
                    edge_counts=per_asset[t]["edge_counts"],
                    robustness_verdict=per_asset[t]["robustness_verdict"],
                    n_folds_5=per_asset[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
        synthetic_perturbation=dict(
            param_sets=grid,
            realistic_medians=sweep_results,
            zero_cost_null_medians=null_results,
            cost_impact=[round(s - n, 3) for s, n in zip(sweep_results, null_results)],
        ),
    )
    
    # Fix the artifact structure - need to extract medians from profile
    for t in ticker_order:
        p = per_asset[t]
        artifact["universe"]["per_asset"][t]["lookback_medians"] = {
            str(lb): p[str(lb)]["medians"] for lb in LOOKBACKS
        }
        artifact["universe"]["per_asset"][t]["lookback_null_medians"] = {
            str(lb): p[str(lb)]["null_medians"] for lb in LOOKBACKS
        }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"  artifact written to: {ARTIFACT_PATH}")

    print("\n=== Cost-robustness check completed ===")
    print("Research / simulation only. No live trading or production execution.")
    print("NOTE: This cost-robustness check used realistic transaction costs and a matched null benchmark.")
    return artifact

if __name__ == "__main__":
    main()
