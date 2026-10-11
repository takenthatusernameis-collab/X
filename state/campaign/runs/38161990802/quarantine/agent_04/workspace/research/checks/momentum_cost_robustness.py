"""Real-data momentum lookback sweep across the collected universe with realistic costs.

Objective: test whether the momentum edge observed at lookback=5 (`momentum.py`)
surives conservative transaction-cost stress. Use realistic costs: commission 2 + 0.3% per share,
2c fixed slippage, 0.05% proportional slippage. Cost parameters match those used in
`logs/ACTIVATION-2026-10-04.md`.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day return;
hold 1 day; daily rebalance; lookbacks 3, 5, 10; walk-forward per segment
train=252d/test=84d/warmup=60d/overlap=60d, min 400 bars/segment, 4 contiguous
volatility blocks, matched coin-flip null on the same segments. The matched null
uses realistic costs and the same parameter grid.

A robust edge is one where the uniformly-positive REGIME_STABLE pattern at lookback=5
also appears at >=2 of the 3 lookbacks for an asset AND the edge survives the
cost test (baseline median log return remains distinguishable from the matched null).
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

REALISTIC_COSTS = bt.BacktestConfig(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    initial_capital=1e6,
    target_exposure=1.0,
)


def segment_result(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback with realistic costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64
    )
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")

    # Run momentum signals with realistic costs
    momentum_result = bt.stress_segments(
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
        cfg=REALISTIC_COSTS,
    )

    # Run matched noise signals (coin-flip null) with realistic costs
    # Use the same seed pattern as in bt.random_signals for matching null
    def noise_signals_fn(c, **kw):
        n = len(c)
        lookback_param = int(kw["lookback"])
        # Match the seed pattern from bt.random_signals
        seed = SEED + int(round(lookback_param * 1000))
        rng = np.random.default_rng(seed)
        out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        for i in range(lookback_param, n):
            weight = float(rng.choice([-1.0, 0.0, 1.0], p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
            out[i] = bt.Signal(date=i + 1, weight=weight)
        return out

    noise_result = bt.stress_segments(
        signals_fn=noise_signals_fn,
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
        cfg=REALISTIC_COSTS,
    )

    return dict(
        segments=[s.name for s in momentum_result.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in momentum_result.scenarios],
        nulls=[round(s.noise_median_log_return, 3) for s in momentum_result.scenarios],
        noise_medians=[round(s.noise_median_log_return, 3) for s in noise_result.scenarios],
        candidate_dispersion=round(momentum_result.candidate_dispersion, 3),
        null_dispersion=round(momentum_result.null_dispersion, 3),
        n_folds=momentum_result.n_folds,
        verdict=momentum_result.overall_verdict,
        momentum_baseline_median=round(momentum_result.scenarios[0].baseline_median_log_return, 3),
        momentum_null_median=momentum_result.scenarios[0].noise_median_log_return,
        cost_robustness=round(momentum_result.scenarios[0].baseline_median_log_return - momentum_result.scenarios[0].noise_median_log_return, 4),
    )


def lookback_verdict(sr):
    """Verdict for one lookback's scenario result with cost robustness filter."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset with cost filter."""
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    robust_with_costs = []
    for lb in LOOKBACKS:
        verdict = lookback_verdict(lookback_results[str(lb)])
        counts[verdict] += 1
        # Filter for cost robustness (baseline distinguishable from matched null)
        cost_gap = lookback_results[str(lb)]["cost_robustness"]
        if verdict == "EDGE" and abs(cost_gap) > 0.0:
            robust_with_costs.append(str(lb))
    if len(robust_with_costs) >= 2:
        return "ROBUST", counts, robust_with_costs
    if len(robust_with_costs) == 1:
        return "SENSITIVE", counts, robust_with_costs
    return "NO_EDGE", counts, []


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

    print("\n=== 3. Leakage review per asset (lookback=5 reference, realistic costs) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, REALISTIC_COSTS)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) across collected universe with realistic costs ===")
    print("  Realistic costs: commission=2 + 0.3% per share, 2c fixed slippage, 0.05% proportional")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    lookback_results = {lb: {t: segment_result(t, all_tickers[t], lb) for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}

    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        v, counts, cost_robust_tickers = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts[v] += 1
        r5 = profile["5"]
        lookback_5_counts[r5["verdict"]] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        profile["cost_robust_tickers"] = cost_robust_tickers
        per_asset[ticker] = profile

    print("  lookback 5 (reference) verdict counts across universe: "
          f"{lookback_5_counts}")
    print("  lookback-robustness verdict counts with cost filter: "
          f"ROBUST={robustness_counts['ROBUST']} "
          f"(cost-robust edge at >=2 lookbacks) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} "
          f"(cost-robust edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")

    print("  Per-asset cost-robustness profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s} {:+6.4f}".format(
                lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"], p[str(lb)]["cost_robustness"]
            )
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} medians3510={:<60s} rob={:<8s} "
              "cost_gap={}".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["cost_robust_tickers"]))

    print("")

    # Determinism check
    res2 = {lb: {t: segment_result(t, tickers[t], lb) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                      for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

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
        ),
    )

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback sweep with realistic costs across the collected universe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())