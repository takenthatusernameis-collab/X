"""Real-data transaction-cost sensitivity for short-horizon momentum.

The cost grid and classification rule are fixed before execution:
0x, 1x, 2x, and 4x the realistic configuration already used by
``examples/ma_crossover_real_data.py``. The 1x configuration is $2 per trade,
$0.003 per share, 2 cents fixed slippage, and 0.0005 proportional slippage.
Every multiplier is applied to all four fields. The matched coin-flip null uses
the same cost configuration as the candidate.

For each asset, the existing lookback-robustness rule is evaluated at 0x and
4x: ROBUST means a positive REGIME_STABLE edge at at least two of lookbacks
3/5/10; SENSITIVE means exactly one; NO_EDGE means none. An asset is
COST_ROBUST only if a ROBUST baseline remains ROBUST at 4x. COST_PARTIAL means
a baseline edge remains detectable but weakens; COST_ELIMINATED means no
positive REGIME_STABLE edge remains. The universe verdict is COST_ROBUST only
if every asset with a baseline edge survives at 4x, COST_FALSIFIED if none do,
and MIXED otherwise.

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
COST_MULTIPLIERS = (0, 1, 2, 4)
BASELINE_COST = 0
STRESS_COST = 4
BASE_COST = {
    "commission_per_trade": 2.0,
    "commission_per_share": 0.003,
    "slippage_cents": 2.0,
    "slippage_proportional": 0.0005,
}
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
LOOKBACK_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"
SURVIVAL_RULE = (
    "positive REGIME_STABLE edge at >=2 of lookbacks 3/5/10 at both 0x and 4x"
)


def cost_config(multiplier):
    """Return the repository cost model scaled by one grid multiplier."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=BASE_COST["commission_per_trade"] * multiplier,
        commission_per_share=BASE_COST["commission_per_share"] * multiplier,
        slippage_cents=BASE_COST["slippage_cents"] * multiplier,
        slippage_proportional=BASE_COST["slippage_proportional"] * multiplier,
    )


def segment_result(ticker, bars, lookback, multiplier):
    """Walk-forward result for one asset, lookback, and cost multiplier."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError(f"{ticker}: signals {len(signals)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cost_config(multiplier),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    return {
        "segments": [s.name for s in res.scenarios],
        "medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(res.null_dispersion, 3),
        "n_folds": res.n_folds,
        "verdict": res.overall_verdict,
    }


def lookback_verdict(result):
    """Classify one lookback result using the existing positive-edge rule."""
    if result["verdict"] == "REGIME_STABLE" and all(m > 0 for m in result["medians"]):
        return "EDGE"
    if result["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(cost_profile):
    """Apply the pre-existing lookback-robustness rule within one cost level."""
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lookback in LOOKBACKS:
        counts[lookback_verdict(cost_profile[str(lookback)])] += 1
    if counts["EDGE"] >= 2:
        verdict = "ROBUST"
    elif counts["EDGE"] == 1:
        verdict = "SENSITIVE"
    else:
        verdict = "NO_EDGE"
    return verdict, counts


def cost_survival(asset):
    """Classify one asset's baseline edge under the maximum cost stress."""
    baseline = asset["robustness_by_cost"][str(BASELINE_COST)]
    stressed = asset["robustness_by_cost"][str(STRESS_COST)]
    if baseline == "NO_EDGE":
        return "NO_BASELINE_EDGE"
    if baseline == "ROBUST" and stressed == "ROBUST":
        return "COST_ROBUST"
    if stressed != "NO_EDGE":
        return "COST_PARTIAL"
    return "COST_ELIMINATED"


def run_grid(tickers):
    """Evaluate every asset, cost, and lookback in a deterministic order."""
    per_asset = {}
    for ticker in tickers:
        profiles = {}
        robustness = {}
        edge_counts = {}
        for multiplier in COST_MULTIPLIERS:
            key = str(multiplier)
            profile = {
                str(lookback): segment_result(
                    ticker, tickers[ticker], lookback, multiplier
                )
                for lookback in LOOKBACKS
            }
            verdict, counts = robustness_verdict(profile)
            profiles[key] = profile
            robustness[key] = verdict
            edge_counts[key] = counts
        asset = {
            "cost_profiles": profiles,
            "robustness_by_cost": robustness,
            "edge_counts_by_cost": edge_counts,
        }
        asset["cost_survival"] = cost_survival(asset)
        per_asset[ticker] = asset
        print(
            "  {:6s} zero={:<8s} 1x={:<8s} 2x={:<8s} 4x={:<8s} -> {}".format(
                ticker,
                robustness["0"],
                robustness["1"],
                robustness["2"],
                robustness["4"],
                asset["cost_survival"],
            )
        )
    return per_asset


def universe_summary(per_asset, ticker_order):
    """Aggregate the fixed cost-survival classification."""
    eligible = [
        t for t in ticker_order
        if per_asset[t]["robustness_by_cost"][str(BASELINE_COST)] != "NO_EDGE"
    ]
    survivors = [
        t for t in eligible
        if per_asset[t]["cost_survival"] == "COST_ROBUST"
    ]
    if not eligible:
        verdict = "NO_BASELINE_EDGE"
    elif len(survivors) == len(eligible):
        verdict = "COST_ROBUST"
    elif not survivors:
        verdict = "COST_FALSIFIED"
    else:
        verdict = "MIXED"
    counts = {"COST_ROBUST": 0, "COST_PARTIAL": 0, "COST_ELIMINATED": 0,
              "NO_BASELINE_EDGE": 0}
    for ticker in ticker_order:
        counts[per_asset[ticker]["cost_survival"]] += 1
    robustness_counts = {}
    for multiplier in COST_MULTIPLIERS:
        key = str(multiplier)
        level = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
        for ticker in ticker_order:
            level[per_asset[ticker]["robustness_by_cost"][key]] += 1
        robustness_counts[key] = level
    return {
        "ticker_order": ticker_order,
        "baseline_assets": eligible,
        "cost_robust_assets": survivors,
        "cost_eliminated_assets": [
            t for t in eligible if per_asset[t]["cost_survival"] == "COST_ELIMINATED"
        ],
        "cost_survival_counts": counts,
        "robustness_counts_by_cost": robustness_counts,
        "universe_verdict": verdict,
    }


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    if pf.main() != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Leakage and fill-equity gate at 4x costs ===")
    tickers = {entry["ticker"]: bt.load_ticker(entry["ticker"])[0]
               for entry in manifest["entries"]}
    for ticker in ["AMZN", "JPM"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        for lookback in LOOKBACKS:
            signals = bt.momentum_signals(closes, lookback=lookback)
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
            res = bt.run_bars(list(bars), signals, cost_config(STRESS_COST))
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print(f"  {ticker}: {bars.n_bars} bars, lookbacks {LOOKBACKS}, leakage [PASS], fill-equity [PASS]")
    print("  no-look-ahead: each signal uses closes[:t] only; costs affect fills, not signals.")

    print("\n=== 4. Cost grid (0x / 1x / 2x / 4x) ===")
    print(f"  1x config: {BASE_COST}")
    print("  The same multiplier is applied to every cost field and to the matched null.")
    print(f"  Fixed classification: {SURVIVAL_RULE}.\n")

    ticker_order = list(tickers.keys())
    per_asset = run_grid(tickers)
    summary = universe_summary(per_asset, ticker_order)

    print("\n=== 5. Universe cost-robustness verdict ===")
    print(f"  baseline assets: {summary['baseline_assets']}")
    print(f"  COST_ROBUST at 4x: {summary['cost_robust_assets']}")
    print(f"  COST_ELIMINATED at 4x: {summary['cost_eliminated_assets']}")
    print(f"  survival counts: {summary['cost_survival_counts']}")
    print(f"  verdict: {summary['universe_verdict']}")

    print("\n=== 6. Zero-cost cross-consistency vs momentum_lookback_sweep_results.json ===")
    with open(LOOKBACK_ARTIFACT) as f:
        previous = json.load(f)
    mismatches = []
    for ticker in ticker_order:
        old = previous["universe"]["per_asset"][ticker]
        old_full = previous["per_asset"][ticker]
        for lookback in LOOKBACKS:
            key = str(lookback)
            new = per_asset[ticker]["cost_profiles"]["0"][key]
            if new["medians"] != old["lookback_medians"][key]:
                mismatches.append(f"{ticker} lookback {lookback} medians")
            if new["null_medians"] != old["lookback_null_medians"][key]:
                mismatches.append(f"{ticker} lookback {lookback} null medians")
            if lookback == 5 and new["verdict"] != old_full[key]["verdict"]:
                mismatches.append(f"{ticker} lookback 5 verdict")
    if mismatches:
        print("  MISMATCH: " + "; ".join(mismatches))
        sys.exit(1)
    print("  all 10 tickers and all 3 lookbacks MATCH the prior zero-cost medians/nulls; lookback-5 verdicts MATCH")

    print("\n=== 7. Determinism ===")
    for ticker in ["AMZN", "JPM"]:
        for multiplier in COST_MULTIPLIERS:
            for lookback in LOOKBACKS:
                first = segment_result(ticker, tickers[ticker], lookback, multiplier)
                second = segment_result(ticker, tickers[ticker], lookback, multiplier)
                if json.dumps(first, sort_keys=True) != json.dumps(second, sort_keys=True):
                    raise AssertionError(f"non-deterministic output for {ticker}, {multiplier}x, lookback {lookback}")
    print("  AMZN/JPM: every cost x lookback cell reproduced exactly")

    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "cost_multipliers": list(COST_MULTIPLIERS),
        "base_cost": BASE_COST,
        "baseline_cost_multiplier": BASELINE_COST,
        "stress_cost_multiplier": STRESS_COST,
        "survival_rule": SURVIVAL_RULE,
        "train": TRAIN,
        "test": TEST,
        "warmup": WARM,
        "overlap": OVERLAP,
        "window": WINDOW,
        "n_blocks": N_BLOCKS,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "per_asset": per_asset,
        "universe": summary,
    }
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())