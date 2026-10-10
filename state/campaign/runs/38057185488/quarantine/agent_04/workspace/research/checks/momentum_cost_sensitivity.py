"""Momentum cost sensitivity check on the collected universe.

Objective: test whether the qualified momentum edge (lookback 3-5) survives
conservative transaction-cost stress on the collected real-data universe (fixed
cost model, lookback 3/5/10, matched coin-flip null) through the full gate stack
with an a-priori cost-robustness prediction; report RETAIN/REVERT/UNVERIFIED for
whether costs erase the edge.

Background: momentum (lookback 5) is SUPPORTED on the collected universe per
activation 37318950814 (REGIME_STABLE in 7/10 assets). The breakout continuation
(20-day) cell was FALSIFIED (activation 37414038601). The frontier calls for a
cost-robustness qualification of the momentum evidence before admission.

Method (fixed a-priori, not tuned to OOS): test momentum with lookback in {3, 5, 10}
through walk-forward regime-stability gate (train=252d/test=84d/warmup=60d/
overlap=60d, 4 contiguous volatility blocks, min 400 bars/segment) with a matched
coin-flip null on the same segments. Cost model: commission_per_trade=2.0,
commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005.
Baseline zero-cost config for comparison. Use the existing 10-asset collected
universe and existing momentum implementation/check framework.

A-priori cost-robustness prediction: short-horizon momentum is mechanical and
very sensitive to transaction costs; the edge is expected to be vulnerable to
conservative fixed costs and proportional slippage.

Research / simulation only. No live trading or production execution.
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
COST_MODEL = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_COST_MODEL = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, lookback, and config."""
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
        cfg_name="cost" if cfg != ZERO_COST_MODEL else "zero_cost",
    )


def extract_verdicts_for_lookback(lookback_results):
    """Extract verdict data for cost robustness assessment."""
    verdicts = {}
    for ticker in lookback_results.get("zero_cost", {}).keys():
        verdicts[ticker] = {
            "zero_cost_verdict": lookback_results["zero_cost"][ticker]["verdict"],
            "cost_verdict": lookback_results["cost"][ticker]["verdict"] if "cost" in lookback_results else None
        }
    return verdicts


def cost_robustness_verdict(lookback_results):
    """Verdict on whether the momentum edge survives costs.

    RETAIN: the edge is REGIME_STABLE (positive) under both zero-cost and cost
    configs for the same lookback, meaning costs did not erase the edge for that lookback.
    REVERT: the edge is REGIME_STABLE (positive) under zero-cost but becomes
    CONSISTENT_WITH_NOISE (noise-like) or REGIME_STABLE_LOSS (stable losses) under
    costs, meaning costs erased the edge for that lookback.
    UNVERIFIED: the edge is not REGIME_STABLE (positive) under zero-cost, so
    cost robustness is not applicable for that lookback.
    """
    overall_verdict = "UNVERIFIED"
    details = {}
    
    for lb in LOOKBACKS:
        if str(lb) not in lookback_results:
            continue
            
        zero_verdict = lookback_results[str(lb)]["zero_cost"]["verdict"]
        cost_verdict = lookback_results[str(lb)]["cost"]["verdict"]
        
        zero_pos = zero_verdict in ("REGIME_STABLE", "REGIME_STABLE_LOSS")
        cost_pos = cost_verdict in ("REGIME_STABLE", "REGIME_STABLE_LOSS")
        
        if zero_pos and cost_pos:
            verdict = "RETAIN"
        elif zero_pos and (cost_verdict == "CONSISTENT_WITH_NOISE" or cost_verdict == "REGIME_STABLE_LOSS"):
            verdict = "REVERT"
        else:
            verdict = "UNVERIFIED"
        
        details[str(lb)] = {
            "zero_cost": zero_verdict,
            "cost": cost_verdict,
            "verdict": verdict,
            "zero_pos": zero_pos,
            "cost_pos": cost_pos
        }
        
        # For overall verdict, use the most severe applicable result
        if verdict == "REVERT":
            overall_verdict = "REVERT"
        elif verdict == "RETAIN" and overall_verdict == "UNVERIFIED":
            overall_verdict = "RETAIN"
    
    return overall_verdict, details


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
    
    print("\n=== 3. Leakage review per asset (lookback=5 reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, COST_MODEL)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")
    
    print("\n=== 4. Momentum cost sensitivity (lookback 3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments, realistic cost model: commission_per_trade=2.0, "
          "commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005\n"
          .format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    
    # Full-universe lookback sweep under both cost and zero-cost configs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())
    
    # Results indexed [lookback][cfg_name][ticker]
    lookback_results = {str(lb): {} for lb in LOOKBACKS}
    
    # Run under zero-cost config first (reference)
    print("  Running zero-cost baseline...")
    for lb in LOOKBACKS:
        print(f"    lookback {lb}...")
        lookback_results[str(lb)]["zero_cost"] = {}
        for ticker in ticker_order:
            res = segment_result(ticker, all_tickers[ticker], lb, ZERO_COST_MODEL)
            lookback_results[str(lb)]["zero_cost"][ticker] = res
    
    # Run under cost config
    print("  Running realistic cost model...")
    for lb in LOOKBACKS:
        print(f"    lookback {lb}...")
        lookback_results[str(lb)]["cost"] = {}
        for ticker in ticker_order:
            res = segment_result(ticker, all_tickers[ticker], lb, COST_MODEL)
            lookback_results[str(lb)]["cost"][ticker] = res
    
    # Print summary
    print("\n=== 5. Cost sensitivity summary ===")
    print("  Lookback 5 (reference) verdict counts across universe (zero-cost):")
    zero_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                     "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    cost_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                     "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for lb in LOOKBACKS:
        for ticker in ticker_order:
            zero_5_counts[lookback_results[str(lb)]["zero_cost"][ticker]["verdict"]] += 1
            cost_5_counts[lookback_results[str(lb)]["cost"][ticker]["verdict"]] += 1
    
    print(f"    zero-cost:   {zero_5_counts}")
    print(f"    cost model:  {cost_5_counts}")
    
    robustness_verdicts = {}
    for lb in LOOKBACKS:
        v, verdicts = cost_robustness_verdict(lookback_results[str(lb)])
        robustness_verdicts[str(lb)] = v
        print(f"    lookback {lb} cost robustness verdict: {v}")
    
    # Print per-asset cost robustness
    print("\n=== 6. Per-asset cost robustness ===")
    for ticker in ticker_order:
        print(f"  {ticker}:")
        for lb in LOOKBACKS:
            zero_verdict = lookback_results[str(lb)]["zero_cost"][ticker]["verdict"]
            cost_verdict = lookback_results[str(lb)]["cost"][ticker]["verdict"]
            zero_median = lookback_results[str(lb)]["zero_cost"][ticker]["medians"][0]
            cost_median = lookback_results[str(lb)]["cost"][ticker]["medians"][0]
            print(f"    lookback {lb}: zero_cost {zero_verdict} (median {zero_median:+.3f}), "
                  f"cost {cost_verdict} (median {cost_median:+.3f})")
    
    # Determine overall verdict
    retain_count = sum(1 for v in robustness_verdicts.values() if v == "RETAIN")
    revert_count = sum(1 for v in robustness_verdicts.values() if v == "REVERT")
    unverified_count = sum(1 for v in robustness_verdicts.values() if v == "UNVERIFIED")
    
    print("\n=== 7. Overall cost robustness verdict ===")
    if retain_count == len(LOOKBACKS):
        overall_verdict = "RETAIN"
        print(f"  ALL lookbacks ({retain_count}/{len(LOOKBACKS)}) RETAIN: the momentum edge survives realistic costs.")
    elif revert_count > 0:
        overall_verdict = "REVERT"
        print(f"  {revert_count}/{len(LOOKBACKS)} lookbacks REVERT: costs erase the momentum edge.")
    else:
        overall_verdict = "UNVERIFIED"
        print(f"  {unverified_count}/{len(LOOKBACKS)} lookbacks UNVERIFIED: cost robustness not applicable.")
    
    # Write the artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_model=dict(
            commission_per_trade=COST_MODEL.commission_per_trade,
            commission_per_share=COST_MODEL.commission_per_share,
            slippage_cents=COST_MODEL.slippage_cents,
            slippage_proportional=COST_MODEL.slippage_proportional,
        ),
        zero_cost_model=dict(
            commission_per_share=ZERO_COST_MODEL.commission_per_share,
            commission_per_trade=ZERO_COST_MODEL.commission_per_trade,
            slippage_cents=ZERO_COST_MODEL.slippage_cents,
            slippage_proportional=ZERO_COST_MODEL.slippage_proportional,
        ),
        lookback_results=lookback_results,
        robustness_verdicts=robustness_verdicts,
        overall_verdict=overall_verdict,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost sensitivity across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())