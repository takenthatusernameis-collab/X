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

# Cost model parameters from examples
COST_ZERO = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
)

COST_REALISTIC = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker and one lookback."""
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
    )


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

    print("\n=== 4. Momentum lookback sweep across collected universe (zero cost) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep for zero cost.
    all_tickers_zero = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers_zero[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers_zero.keys())

    # Results indexed [lookback][ticker] for zero cost.
    lookback_results_zero = {lb: {t: segment_result(t, all_tickers_zero[t], lb, COST_ZERO) for t in ticker_order}
                             for lb in LOOKBACKS}

    # Process zero-cost results.
    per_asset_zero = {}
    robustness_counts_zero = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts_zero = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                              "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        profile = {}
        for lb in LOOKBACKS:
            profile[str(lb)] = lookback_results_zero[lb][ticker]
        
        # Count lookback robustness.
        counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            verdict = profile[str(lb)]["verdict"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in profile[str(lb)]["medians"]):
                counts["EDGE"] += 1
            elif verdict == "REGIME_STABLE_LOSS":
                counts["LOSS"] += 1
            else:
                counts["NO_EDGE"] += 1

        if counts["EDGE"] >= 2:
            v = "ROBUST"
            robustness_counts_zero[v] += 1
        elif counts["EDGE"] == 1:
            v = "SENSITIVE"
            robustness_counts_zero[v] += 1
        else:
            v = "NO_EDGE"
            robustness_counts_zero[v] += 1

        r5 = profile["5"]
        lookback_5_counts_zero[r5["verdict"]] += 1

        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        per_asset_zero[ticker] = profile

    print("  Lookback 5 (zero cost) verdict counts across universe: "
          f"{lookback_5_counts_zero}")
    print("  Lookback-robustness verdict counts (zero cost): "
          f"ROBUST={robustness_counts_zero['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts_zero['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts_zero['NO_EDGE']}\n")

    print("\n=== 5. Momentum lookback sweep across collected universe (realistic costs) ===")
    # Full-universe lookback sweep for realistic costs.
    all_tickers_realistic = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers_realistic[ticker] = bt.load_ticker(ticker)[0]

    # Results indexed [lookback][ticker] for realistic costs.
    lookback_results_realistic = {lb: {t: segment_result(t, all_tickers_realistic[t], lb, COST_REALISTIC) for t in ticker_order}
                                  for lb in LOOKBACKS}

    # Process realistic cost results.
    per_asset_realistic = {}
    robustness_counts_realistic = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts_realistic = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                                  "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        profile = {}
        for lb in LOOKBACKS:
            profile[str(lb)] = lookback_results_realistic[lb][ticker]
        
        # Count lookback robustness.
        counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            verdict = profile[str(lb)]["verdict"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in profile[str(lb)]["medians"]):
                counts["EDGE"] += 1
            elif verdict == "REGIME_STABLE_LOSS":
                counts["LOSS"] += 1
            else:
                counts["NO_EDGE"] += 1

        if counts["EDGE"] >= 2:
            v = "ROBUST"
            robustness_counts_realistic[v] += 1
        elif counts["EDGE"] == 1:
            v = "SENSITIVE"
            robustness_counts_realistic[v] += 1
        else:
            v = "NO_EDGE"
            robustness_counts_realistic[v] += 1

        r5 = profile["5"]
        lookback_5_counts_realistic[r5["verdict"]] += 1

        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        per_asset_realistic[ticker] = profile

    print("  Lookback 5 (realistic cost) verdict counts across universe: "
          f"{lookback_5_counts_realistic}")
    print("  Lookback-robustness verdict counts (realistic cost): "
          f"ROBUST={robustness_counts_realistic['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts_realistic['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts_realistic['NO_EDGE']}\n")

    print("  Cost-robustness summary:")
    robust_assets_zero = sum(1 for ticker in ticker_order if per_asset_zero[ticker]["robustness_verdict"] == "ROBUST")
    robust_assets_realistic = sum(1 for ticker in ticker_order if per_asset_realistic[ticker]["robustness_verdict"] == "ROBUST")
    
    print(f"  Zero-cost: {robust_assets_zero}/{len(ticker_order)} assets ROBUST")
    print(f"  Realistic cost: {robust_assets_realistic}/{len(ticker_order)} assets ROBUST")
    print(f"  Cost-erosion: {robust_assets_zero - robust_assets_realistic} assets lost edge under realistic costs")
    
    if robust_assets_realistic > 0:
        print(f"  CONCLUSION: Momentum edge survives realistic transaction costs on {robust_assets_realistic} assets")
    else:
        print(f"  CONCLUSION: Momentum edge is cost-eroded; no assets maintain REGIME_STABLE positive edge under realistic costs")

    # Construct artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=dict(
            zero_cost=per_asset_zero,
            realistic_cost=per_asset_realistic,
        ),
        universe=dict(
            lookback_5_counts_zero=lookback_5_counts_zero,
            lookback_5_counts_realistic=lookback_5_counts_realistic,
            robustness_counts_zero=robustness_counts_zero,
            robustness_counts_realistic=robustness_counts_realistic,
            ticker_order=ticker_order,
        ),
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print("\n=== Artifact written to: ===")
    print(f"  {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost sensitivity on the collected universe: exploratory simulation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
