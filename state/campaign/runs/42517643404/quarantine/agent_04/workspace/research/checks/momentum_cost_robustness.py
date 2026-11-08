#!/usr/bin/env python3
"""Cost-sensitivity check for momentum lookback 3/5/10 on the collected universe.

Objective: test whether the short-horizon momentum edge survives conservative
transaction-cost stress. The existing momentum results show an edge at lookback=5
in many assets, but cost robustness is unknown and decision-relevant.

Method (fixed a-priori, not tuned to OOS):
- long the previous N-day return; hold 1 day; daily rebalance
- walk-forward per segment train=252d / test=84d / warmup=60d / overlap=60d
- zero cost (cfg0) and realistic costs (cfg1) as defined in the repository
- compare OOS log returns across 10 collected tickers
- unmatched null is zero cost for reference
- preserve exact repository cost model and parameter grids
- deterministic recomputation required

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Cost configurations matching repository examples
CFG0 = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)

CFG1 = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness_results.json"


def momentum_signals(closes: np.ndarray, lookback: int):
    """Momentum signal implementation (matches existing momentum_signals)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, one cfg."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = momentum_signals(closes, lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=momentum_signals,
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
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Manifest integrity
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"
    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | universe: {len(manifest['universe'])}")

    # 2. Data preflight gate
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    # 3. Load all tickers from manifest
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    print(f"\n=== 3. Cost-sensitivity sweep (lookback {LOOKBACKS}) ===")
    print("  Config 0 (zero cost): no commissions, no slippage")
    print("  Config 1 (realistic costs): commission_per_trade=2.0, commission_per_share=0.003,")
    print("                             slippage_cents=2.0, slippage_proportional=0.0005")
    print(f"  Walk-forward: train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d")

    # 4. Run cost-sensitivity analysis for each lookback
    results = {}
    for lb in LOOKBACKS:
        print(f"\n--- Lookback = {lb} ---")
        results[lb] = {
            "zero_cost": {},  # per asset results with zero costs
            "realistic_cost": {},  # per asset results with realistic costs
        }
        for ticker in ticker_order:
            bars = all_tickers[ticker]
            # Zero cost run
            print(f"  {ticker} lookback={lb}: zero cost ...")
            results[lb]["zero_cost"][ticker] = segment_result(ticker, bars, lb, CFG0)
            # Realistic cost run
            print(f"  {ticker} lookback={lb}: realistic cost ...")
            results[lb]["realistic_cost"][ticker] = segment_result(ticker, bars, lb, CFG1)

    # 5. Summarize results
    print("\n=== 4. Summary ===")
    print("  0.0 = zero cost | 1.0 = realistic cost")
    print("  each row: zero_cost_median realistic_cost_median  survival_ratio")
    print("  survival_ratio = realistic_cost_median / zero_cost_median (if zero_cost != 0)")

    # Track verdict changes and survival patterns
    verdict_changes = {lb: {"REGIME_STABLE": 0, "REGIME_DEPENDENT": 0, "CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE_LOSS": 0} for lb in LOOKBACKS}
    survival_ratios = {}

    for lb in LOOKBACKS:
        zero_cost_medians = []
        realistic_cost_medians = []
        zero_cost_verdicts = []
        realistic_cost_verdicts = []

        for ticker in ticker_order:
            zc = results[lb]["zero_cost"][ticker]
            rc = results[lb]["realistic_cost"][ticker]
            zero_cost_medians.append(zc["medians"][0])
            realistic_cost_medians.append(rc["medians"][0])
            zero_cost_verdicts.append(zc["verdict"])
            realistic_cost_verdicts.append(rc["verdict"])

        # Print per-asset comparison
        for i, ticker in enumerate(ticker_order):
            zc = results[lb]["zero_cost"][ticker]
            rc = results[lb]["realistic_cost"][ticker]
            zero_median = zc["medians"][0]
            realistic_median = rc["medians"][0]
            if abs(zero_median) > 1e-12:
                survival = realistic_median / zero_median
                survival_ratios[lb] = survival
            elif realistic_median != 0:
                survival = "inf"
                survival_ratios[lb] = survival
            else:
                survival = "N/A"
                survival_ratios[lb] = survival
            verdict_changes[lb][zc["verdict"]] += 1
            verdict_changes[lb][rc["verdict"]] += 1

            print(f"  {ticker:<6s}: zc={zero_median:+.3f} rc={realistic_median:+.3f}  verdict zc={zc['verdict']:<20s} rc={rc['verdict']}")

        # Print verdict changes summary
        print(f"\n  Lookback {lb} verdict changes:")
        for verdict, count in verdict_changes[lb].items():
            if count > 0:
                print(f"    {verdict}: {count}")

        # Calculate survival statistics
        valid_pairs = []
        for ticker in ticker_order:
            zc = results[lb]["zero_cost"][ticker]
            rc = results[lb]["realistic_cost"][ticker]
            if abs(zc["medians"][0]) > 1e-12:
                valid_pairs.append((zc, rc))

        if valid_pairs:
            survival_rates = []
            for zc, rc in valid_pairs:
                survival = rc["medians"][0] / zc["medians"][0]
                survival_rates.append(survival)

            survival_rates = [s for s in survival_rates if not np.isinf(s) and not np.isnan(s)]
            if survival_rates:
                mean_survival = np.mean(survival_rates)
                std_survival = np.std(survival_rates, ddof=1) if len(survival_rates) > 1 else 0.0
                print(f"  Mean survival ratio: {mean_survival:.3f} ± {std_survival:.3f}")

    # 6. Determinism check (rerun target subset)
    print("\n=== 5. Determinism check ===")
    target = ["AMZN", "JPM", "AAPL", "MSFT"]
    determinism_ok = True

    for lb in LOOKBACKS:
        res2 = {}
        for ticker in target:
            bars = all_tickers[ticker]
            res2[ticker] = segment_result(ticker, bars, lb, CFG1)

        # Compare with first run results
        res1 = results[lb]["realistic_cost"]
        out1 = json.dumps({t: res1[t] for t in target}, sort_keys=True)
        out2 = json.dumps(res2, sort_keys=True)
        match = out1 == out2
        if not match:
            determinism_ok = False
            print(f"  Lookback {lb} DETERMINISM FAIL: results differ between runs")
        else:
            print(f"  Lookback {lb} determinism: OK")

    if not determinism_ok:
        sys.exit(1)

    # 7. Write artifact for independent verification
    artifact = dict(
        dataset_id=DATASET_ID,
        lookbacks=list(LOOKBACKS),
        seed=SEED,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        results=results,
        verdict_changes=verdict_changes,
        survival_ratios=survival_ratios,
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\n=== 6. Artifact written to {ARTIFACT_PATH} ===")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost-robustness check across the collected universe: exploratory simulation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())