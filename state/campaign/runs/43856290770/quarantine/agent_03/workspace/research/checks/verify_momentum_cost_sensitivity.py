"""Independent verification of momentum cost sensitivity.

This verifier does not call ``stress_segments`` or import the check module. It
rebuilds volatility blocks and contiguous segments, creates momentum signals,
and recomputes candidate medians with direct ``walk_forward`` calls. Matched
nulls are recomputed through ``noise_benchmark`` using the same BacktestConfig.
Every asset, cost multiplier, lookback, median, null median, dispersion, fold
count, verdict, robustness classification, and universe aggregate is compared.

The verifier also asserts deterministic recomputation of a 4x AMZN cell.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
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
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"


def config(multiplier):
    """Reconstruct the scaled repository cost configuration."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=BASE_COST["commission_per_trade"] * multiplier,
        commission_per_share=BASE_COST["commission_per_share"] * multiplier,
        slippage_cents=BASE_COST["slippage_cents"] * multiplier,
        slippage_proportional=BASE_COST["slippage_proportional"] * multiplier,
    )


def blocks(closes):
    """Independent contiguous volatility-block construction."""
    n = len(closes)
    block_size = n // N_BLOCKS
    bar_vols = []
    for i in range(n):
        if i < WINDOW:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - WINDOW:i]), ddof=1)) * np.sqrt(252.0)
            )
    active = [v for v in bar_vols if v > 0]
    series_median = float(np.median(active))
    labels = []
    for i in range(n):
        if i < WINDOW:
            labels.append("insufficient")
            continue
        block = i // block_size
        start, end = block * block_size, (block + 1) * block_size
        if end - start <= WINDOW:
            labels.append("calm")
            continue
        block_values = [v for v in bar_vols[start:end] if v > 0]
        block_median = float(np.median(block_values))
        labels.append("turbulent" if block_median > series_median else "calm")
    return labels


def segments(labels):
    """Independent contiguous-run reconstruction."""
    runs = []
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            if i - start >= MIN_SEGMENT_BARS:
                runs.append((current, start, i))
            current = labels[i]
            start = i
    if len(labels) - start >= MIN_SEGMENT_BARS:
        runs.append((current, start, len(labels)))
    return runs


def signals(closes, lookback):
    """Independent momentum signal construction."""
    result = [bt.Signal(date=i + 1, weight=0.0) for i in range(len(closes))]
    for i in range(int(lookback), len(closes)):
        mean_log = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
        result[i] = bt.Signal(date=i + 1, weight=1.0 if mean_log > 0.0 else -1.0)
    return result


def regime_verdict(medians, null_medians):
    """Re-derive the repository verdict from unrounded values."""
    candidate = np.asarray(medians, dtype=np.float64)
    null = np.asarray(null_medians, dtype=np.float64)
    if all(abs(value) <= 0.05 for value in candidate):
        return "CONSISTENT_WITH_NOISE"
    if float(np.std(candidate)) > 2.0 * float(np.std(null)):
        return "REGIME_DEPENDENT"
    if all(value < 0.0 for value in candidate):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def cell(ticker, bars, lookback, multiplier):
    """Fresh direct-path recomputation for one asset/cost/lookback cell."""
    closes = bars.closes_array()
    labels = blocks(closes)
    cfg = config(multiplier)
    sig = signals(closes, lookback)
    candidate = []
    null = []
    segment_names = []
    fold_count = None
    for name, start, end in segments(labels):
        segment_bars = list(bars)[start:end]
        segment_signals = sig[start:end]
        result = bt.walk_forward(
            segment_bars,
            segment_signals,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=cfg,
        )
        returns = np.asarray(
            [fold.metrics["total_return"] for fold in result.folds], dtype=np.float64
        )
        log_returns = np.log1p(np.clip(returns, -1.0 + 1e-12, None))
        candidate.append(float(np.median(log_returns)))
        noise = noise_benchmark(
            bars=segment_bars,
            param_grid=[{"lookback": lookback}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=0,
            overlap_window=0,
            cfg=cfg,
            periods_per_year=252,
        )
        null.append(float(noise.baseline_median_log_return))
        segment_names.append(name)
        if fold_count is None:
            fold_count = len(result.folds)
    return {
        "segments": segment_names,
        "medians": [round(value, 3) for value in candidate],
        "null_medians": [round(value, 3) for value in null],
        "candidate_dispersion": round(float(np.std(np.asarray(candidate))), 3),
        "null_dispersion": round(float(np.std(np.asarray(null))), 3),
        "n_folds": fold_count,
        "verdict": regime_verdict(candidate, null),
    }


def edge_kind(result):
    """Classify one lookback using only positive REGIME_STABLE medians."""
    if result["verdict"] == "REGIME_STABLE" and all(m > 0 for m in result["medians"]):
        return "EDGE"
    if result["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness(profile):
    """Re-derive the lookback-robustness count and label."""
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lookback in LOOKBACKS:
        counts[edge_kind(profile[str(lookback)])] += 1
    if counts["EDGE"] >= 2:
        verdict = "ROBUST"
    elif counts["EDGE"] == 1:
        verdict = "SENSITIVE"
    else:
        verdict = "NO_EDGE"
    return verdict, counts


def survival(asset):
    """Re-derive the predeclared asset-level cost-survival class."""
    baseline = asset["robustness_by_cost"][str(BASELINE_COST)]
    stressed = asset["robustness_by_cost"][str(STRESS_COST)]
    if baseline == "NO_EDGE":
        return "NO_BASELINE_EDGE"
    if baseline == "ROBUST" and stressed == "ROBUST":
        return "COST_ROBUST"
    if stressed != "NO_EDGE":
        return "COST_PARTIAL"
    return "COST_ELIMINATED"


def main():
    np.random.seed(42)
    with open(ARTIFACT) as f:
        published = json.load(f)
    assert published["dataset_id"] == DATASET_ID
    assert published["lookbacks"] == list(LOOKBACKS)
    assert published["cost_multipliers"] == list(COST_MULTIPLIERS)
    assert published["base_cost"] == BASE_COST

    bars_by_ticker = {
        ticker: bt.load_ticker(ticker)[0]
        for ticker in published["universe"]["ticker_order"]
    }

    print("=== Gate 1: independent direct walk_forward + matched null recomputation ===")
    fresh_assets = {}
    all_ok = True
    for ticker, bars in bars_by_ticker.items():
        profiles = {}
        robustness_by_cost = {}
        edge_counts_by_cost = {}
        for multiplier in COST_MULTIPLIERS:
            key = str(multiplier)
            profile = {}
            for lookback in LOOKBACKS:
                recomputed = cell(ticker, bars, lookback, multiplier)
                expected = published["per_asset"][ticker]["cost_profiles"][key][str(lookback)]
                match = recomputed == expected
                if not match:
                    all_ok = False
                    print(
                        f"  MISMATCH {ticker} cost={multiplier}x lookback={lookback}: "
                        f"recomputed={recomputed} published={expected}"
                    )
                profile[str(lookback)] = recomputed
            verdict, counts = robustness(profile)
            profiles[key] = profile
            robustness_by_cost[key] = verdict
            edge_counts_by_cost[key] = counts
        asset = {
            "cost_profiles": profiles,
            "robustness_by_cost": robustness_by_cost,
            "edge_counts_by_cost": edge_counts_by_cost,
        }
        asset["cost_survival"] = survival(asset)
        fresh_assets[ticker] = asset
        expected_asset = published["per_asset"][ticker]
        if asset != expected_asset:
            all_ok = False
            print(f"  MISMATCH {ticker}: recomputed asset classification differs")
        else:
            print(
                f"  {ticker}: all 12 cells MATCH; "
                f"0x={robustness_by_cost['0']} 4x={robustness_by_cost['4']} "
                f"-> {asset['cost_survival']}"
            )

    print("\n=== Gate 2: universe aggregate and survival verdict ===")
    eligible = [
        ticker for ticker in bars_by_ticker
        if fresh_assets[ticker]["robustness_by_cost"][str(BASELINE_COST)] != "NO_EDGE"
    ]
    survivors = [
        ticker for ticker in eligible
        if fresh_assets[ticker]["cost_survival"] == "COST_ROBUST"
    ]
    if not eligible:
        universe_verdict = "NO_BASELINE_EDGE"
    elif len(survivors) == len(eligible):
        universe_verdict = "COST_ROBUST"
    elif not survivors:
        universe_verdict = "COST_FALSIFIED"
    else:
        universe_verdict = "MIXED"
    survival_counts = {
        "COST_ROBUST": 0,
        "COST_PARTIAL": 0,
        "COST_ELIMINATED": 0,
        "NO_BASELINE_EDGE": 0,
    }
    for asset in fresh_assets.values():
        survival_counts[asset["cost_survival"]] += 1
    robustness_counts = {}
    for multiplier in COST_MULTIPLIERS:
        key = str(multiplier)
        level = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
        for asset in fresh_assets.values():
            level[asset["robustness_by_cost"][key]] += 1
        robustness_counts[key] = level
    recomputed_summary = {
        "ticker_order": list(bars_by_ticker.keys()),
        "baseline_assets": eligible,
        "cost_robust_assets": survivors,
        "cost_eliminated_assets": [
            ticker for ticker in eligible
            if fresh_assets[ticker]["cost_survival"] == "COST_ELIMINATED"
        ],
        "cost_survival_counts": survival_counts,
        "robustness_counts_by_cost": robustness_counts,
        "universe_verdict": universe_verdict,
    }
    if recomputed_summary != published["universe"]:
        all_ok = False
        print(f"  MISMATCH: recomputed summary={recomputed_summary}")
        print(f"            published summary={published['universe']}")
    else:
        print(f"  MATCH: {recomputed_summary}")

    print("\n=== Gate 3: determinism at 4x (AMZN lookback 5) ===")
    bars = bars_by_ticker["AMZN"]
    first = cell("AMZN", bars, 5, STRESS_COST)
    second = cell("AMZN", bars, 5, STRESS_COST)
    deterministic = json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    print(f"  r1={first}")
    print(f"  r2={second}")
    print(f"  identical: {deterministic}")
    if not deterministic:
        all_ok = False

    print()
    if not all_ok:
        print("MISMATCH FOUND: independent recomputation does not match the artifact.")
        return 1
    print("All independent recomputations MATCH the check artifact and universe verdict.")
    print("NOTE: research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())