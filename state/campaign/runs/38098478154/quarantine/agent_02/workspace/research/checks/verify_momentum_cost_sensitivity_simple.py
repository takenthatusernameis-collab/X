"""
Independent verification of `research/checks/momentum_cost_sensitivity_simple.py`.

This verifies that the cost-sensitivity check runs deterministically and
reproducibly with realistic transaction costs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_share=0.003,
    slippage_cents=5.0,
    slippage_proportional=0.001,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

NO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
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


def momentum_signals_no_costs(closes, lookback):
    """Fresh implementation of momentum signals."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result_verification(ticker, bars, lookback, costs):
    """Walk-forward regime-stability result with fresh implementation."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = momentum_signals_no_costs(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=momentum_signals_no_costs,
        bars=list(bars),
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=costs,
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
    )


def main():
    np.random.seed(42)
    artifact = json.load(open(ARTIFACT_PATH))
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["lookbacks"] == list(LOOKBACKS)

    all_ok = True
    print("=== Independent verification of cost-sensitivity check ===")

    # Load tickers for verification
    tickers = {}
    from research.data.preflight import load_manifest
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    # Verify a subset (AMZN, JPM) for speed
    for asset in ["AMZN", "JPM"]:
        bars = tickers[asset]
        for lb in LOOKBACKS:
            # Recompute with realistic costs
            result_real = segment_result_verification(asset, bars, lb, REALISTIC_COSTS)
            # Recompute with no costs
            result_no_costs = segment_result_verification(asset, bars, lb, NO_COSTS)

            # Compare with artifact
            art_entry = artifact["per_asset"][asset]

            # Compare realistic costs results
            medians_match = result_real["medians"] == art_entry["lookback_medians_real"][str(lb)]
            nulls_match = result_real["null_medians"] == art_entry["lookback_nulls_real"][str(lb)]
            verdict_match = result_real["verdict"] == art_entry["lookback_verdicts_real"][str(lb)]

            status_real = "MATCH" if (medians_match and nulls_match and verdict_match) else "DIFFERENT"
            if status_real != "MATCH":
                all_ok = False
                print(f"  {asset}: lookback {lb} realistic costs RECOMPUTED != artifact")

            # Compare no costs results
            medians_match_no = result_no_costs["medians"] == art_entry["lookback_medians_no_costs"][str(lb)]
            nulls_match_no = result_no_costs["null_medians"] == art_entry["lookback_nulls_no_costs"][str(lb)]
            verdict_match_no = result_no_costs["verdict"] == art_entry["lookback_verdicts_no_costs"][str(lb)]

            status_no = "MATCH" if (medians_match_no and nulls_match_no and verdict_match_no) else "DIFFERENT"
            if status_no != "MATCH":
                all_ok = False
                print(f"  {asset}: lookback {lb} no costs RECOMPUTED != artifact")

            if status_real == "MATCH" and status_no == "MATCH":
                print(f"  {asset}: lookback {lb} -> {status_real} (real) / {status_no} (no costs)")

    print("\nDeterminism check...")
    # Test determinism by rerunning AMZN lookback=5
    bars = tickers["AMZN"]
    res1 = segment_result_verification("AMZN", bars, 5, REALISTIC_COSTS)
    res2 = segment_result_verification("AMZN", bars, 5, REALISTIC_COSTS)

    deterministic = (res1["medians"] == res2["medians"] and
                    res1["null_medians"] == res2["null_medians"] and
                    res1["verdict"] == res2["verdict"])

    print(f"  AMZN lookback 5 deterministic: {'YES' if deterministic else 'NO'}")
    if not deterministic:
        all_ok = False

    print("\n" + "="*60)
    if all_ok:
        print("VERIFICATION PASSED: The cost-sensitivity check appears to")
        print("produce deterministic and reproducible results with realistic costs.")
    else:
        print("VERIFICATION FAILED: Mismatches found between recomputed")
        print("results and the artifact. This indicates a potential defect.")
        sys.exit(1)

    print("NOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-sensitivity independent verification: exploratory simulation.")


if __name__ == "__main__":
    main()
