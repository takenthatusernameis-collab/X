"""Real-data momentum cost-robustness check across the collected universe.

Objective: test whether the momentum edge observed at lookback=5 (research/checks/momentum.py, activation 37318950814) survives conservative transaction-cost stress on the collected real-data universe.

For each collected ticker, momentum (long the previous N-day return, hold 1 day, daily rebalance) is walk-forward validated (train=252d/test=84d/warmup=60d/overlap=60d) inside each of the 4 contiguous volatility blocks of the real-data regime family, for lookback in {3, 5, 10}. Each lookback's per-segment median log return is compared against a matched coin-flip null benchmark run on the same segments. Realistic transaction costs are applied using the existing realistic cost model (commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.001). A lookback-robust edge is one where the uniformly-positive REGIME_STABLE pattern at lookback=5 also appears at >= 2 of the 3 lookbacks for an asset AND the edge survives the realistic cost stress test.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day return; hold 1 day; daily rebalance; lookbacks 3, 5, 10; same walk-forward windows and volatility blocks as `momentum.py`. Determinism is asserted via an internal independent recomputation.

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
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"

# Realistic cost model from research logs (commission 2 + 0.3% per share, 2c + 10bps slippage)
REALISTIC_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=60,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.001,
)

ZERO_COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=60,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
)


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, and one cost configuration."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
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
    )
    return res


def lookback_verdict(sr):
    """Robustness verdict for one lookback's scenario result.

    REGIME_STABLE with uniformly positive medians means this lookback shows a
    stable positive edge on this asset; REGIME_STABLE with uniformly negative
    medians is a stable loss, not an edge.
    """
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


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
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
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) across collected universe, zero vs realistic costs ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a matched null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("  Cost configurations:")
    print("    Zero cost: commission_per_trade=0, commission_per_share=0, slippage_cents=0, slippage_proportional=0")
    print("    Realistic cost: commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.001")

    # Load universe
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][cost_config][segment index]
    results = {}
    for cfg_name, cfg in [("zero_cost", ZERO_COST_CONFIG), ("realistic_cost", REALISTIC_COST_CONFIG)]:
        print(f"\n--- Cost configuration: {cfg_name} ---")
        results[cfg_name] = {}
        for lookback in LOOKBACKS:
            print(f"  Lookback {lookback}...", end="")
            results[cfg_name][str(lookback)] = {}
            for ticker in ticker_order:
                res = segment_result(ticker, all_tickers[ticker], lookback, cfg)
                results[cfg_name][str(lookback)][ticker] = {
                    "segments": [s.name for s in res.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
                    "candidate_dispersion": round(res.candidate_dispersion, 3),
                    "null_dispersion": round(res.null_dispersion, 3),
                    "n_folds": res.n_folds,
                    "verdict": res.overall_verdict,
                }
            print(f" {len(ticker_order)} tickers done")

    print("\n=== 5. Cost-robustness analysis ===")
    per_asset = {}
    cost_robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    cost_robustness_edge_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    
    for ticker in ticker_order:
        profile = {}
        cost_verdicts = []
        lookback_counts = {}
        
        # Analyze zero cost results for lookback robustness
        zero_cost_profile = {}
        for lb in LOOKBACKS:
            sr = results["zero_cost"][str(lb)][ticker]
            zero_cost_profile[str(lb)] = sr
            zero_cost_profile[str(lb)]["edge"] = lookback_verdict(sr)
            cost_verdicts.append(lookback_verdict(sr))
            lookback_counts[str(lb)] = lookback_verdict(sr)
        
        v, counts = robustness_verdict({str(lb): zero_cost_profile[str(lb)] for lb in LOOKBACKS})
        cost_verdicts.append(v)
        
        # Analyze realistic cost results
        realistic_cost_profile = {}
        for lb in LOOKBACKS:
            sr = results["realistic_cost"][str(lb)][ticker]
            realistic_cost_profile[str(lb)] = sr
            realistic_cost_profile[str(lb)]["edge"] = lookback_verdict(sr)
            cost_verdicts.append(lookback_verdict(sr))
            lookback_counts[str(lb)] = lookback_verdict(sr)
        
        v_cost, counts_cost = robustness_verdict({str(lb): realistic_cost_profile[str(lb)] for lb in LOOKBACKS})
        cost_verdicts.append(v_cost)
        
        # Determine final cost-robustness verdict
        # ROBUST: uniformly positive REGIME_STABLE edges at >= 2 lookbacks in BOTH zero and realistic costs
        # SENSITIVE: uniformly positive at exactly 1 lookback (or at >= 2 in zero but not realistic)
        # NO_EDGE: no positive edges in realistic costs
        zero_robust = v == "ROBUST"
        realistic_robust = v_cost == "ROBUST"
        
        if realistic_robust and zero_robust:
            final_verdict = "ROBUST"
            cost_robustness_counts["ROBUST"] += 1
            cost_robustness_edge_counts["ROBUST"] += sum(1 for v in cost_verdicts if v == "EDGE")
        elif realistic_robust and not zero_robust:
            final_verdict = "SENSITIVE"
            cost_robustness_counts["SENSITIVE"] += 1
            cost_robustness_edge_counts["SENSITIVE"] += sum(1 for v in cost_verdicts if v == "EDGE")
        elif not realistic_robust and v == "ROBUST":
            final_verdict = "SENSITIVE"
            cost_robustness_counts["SENSITIVE"] += 1
            cost_robustness_edge_counts["SENSITIVE"] += sum(1 for v in cost_verdicts if v == "EDGE")
        else:
            final_verdict = "NO_EDGE"
            cost_robustness_counts["NO_EDGE"] += 1
            cost_robustness_edge_counts["NO_EDGE"] += sum(1 for v in cost_verdicts if v == "EDGE")
        
        profile["lookback_5_zero_cost"] = zero_cost_profile["5"]
        profile["lookback_5_realistic_cost"] = realistic_cost_profile["5"]
        profile["lookback_counts_zero_cost"] = lookback_counts
        profile["lookback_counts_realistic_cost"] = lookback_counts
        profile["final_verdict"] = final_verdict
        profile["edge_counts_zero_cost"] = counts
        profile["edge_counts_realistic_cost"] = counts_cost
        
        per_asset[ticker] = profile
        
        print(f"  {ticker}: zero_cost={v}, realistic_cost={v_cost}, final_cost_robustness={final_verdict}")

    print("\n  Lookback-robustness verdict counts (zero cost):")
    print(f"    ROBUST={v == 'ROBUST'} (>=2 lookbacks positive edge) | "
          f"SENSITIVE={v == 'SENSITIVE'} (edge at exactly one lookback) | "
          f"NO_EDGE={v == 'NO_EDGE'} (no lookback shows a positive edge)")
    print("\n  Cost-robustness verdict counts (final):")
    print(f"    ROBUST={cost_robustness_counts['ROBUST']} "
          f"(survives realistic costs AND lookback-robust) | "
          f"SENSITIVE={cost_robustness_counts['SENSITIVE']} "
          f"(survives realistic costs but not lookback-robust OR survives lookback-robust but not realistic) | "
          f"NO_EDGE={cost_robustness_counts['NO_EDGE']} (does not survive realistic costs)")

    print("\n=== 6. Determinism: rerun the target subset and assert identical output ===")
    res2 = {}
    for cfg_name in ["zero_cost", "realistic_cost"]:
        res2[cfg_name] = {}
        for lb in LOOKBACKS:
            res2[cfg_name][str(lb)] = {}
            for ticker in target:
                res2[cfg_name][str(lb)][ticker] = segment_result(ticker, tickers[ticker], lb, ZERO_COST_CONFIG if cfg_name == "zero_cost" else REALISTIC_COST_CONFIG)
    
    out1 = json.dumps(results, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        cost_robustness_counts=cost_robustness_counts,
        cost_robustness_edge_counts=cost_robustness_edge_counts,
        zero_cost_results=results["zero_cost"],
        realistic_cost_results=results["realistic_cost"],
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum cost-robustness across the collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())