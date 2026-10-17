"""Cost-sensitivity check for lookback 3/5/10 momentum under realistic cost model.

Objective: test whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe (primary question from task R-001).

Method: run the momentum lookback sweep (lookback 3/5/10) with both zero-cost
and realistic-cost models, then compare whether the edge persists. Realistic costs:
- commission_per_trade: $2.0 per round-trip
- commission_per_share: $0.003 per share (0.3%)
- slippage_cents: $0.02 per share
- slippage_proportional: 5 bps (0.0005) per share

The realistic cost model is taken from examples/ma_crossover.py cfg1.

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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def segment_result_with_costs(
    ticker: str,
    bars,
    lookback: int,
    cost_config: bt.BacktestConfig,
):
    """Walk-forward regime-stability result for one ticker and one lookback with given costs."""
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
        cfg=cost_config,
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
    """Verdict for one lookback's scenario result.

    REGIME_STABLE with uniformly positive medians means this lookback shows a
    stable positive edge on this asset; REGIME_STABLE_LOSS is a stable loss,
    not an edge.
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    # Cost configurations: zero cost and realistic costs
    cfg_zero = bt.BacktestConfig(initial_capital=1e6, warmup_periods=0)
    cfg_realistic = bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=0,
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )

    print("\n=== 3. Leakage review per asset (lookback=5 reference, zero cost) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res_zero = bt.run_bars(list(bars), signals, cfg_zero)
        bt.check_equity_matches_fills(res_zero.equity_curve, res_zero.trades, closes, 1e6)
        total_return = (res_zero.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker} (zero cost): {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
        res_real = bt.run_bars(list(bars), signals, cfg_realistic)
        total_return_real = (res_real.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker} (realistic cost): total_return {total_return_real:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep with zero cost
    print("\n  === Zero cost baseline ===")
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    zero_lookback_results = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, cfg_zero) for t in ticker_order}
                             for lb in LOOKBACKS}

    zero_per_asset = {}
    zero_robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        profile = {str(lb): zero_lookback_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        zero_robustness_counts[v] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        zero_per_asset[ticker] = profile

    print("  lookback-robustness verdict counts (zero cost):")
    print(f"    ROBUST={zero_robustness_counts['ROBUST']} (>=2 lookbacks positive edge)")
    print(f"    SENSITIVE={zero_robustness_counts['SENSITIVE']} (edge at exactly one lookback)")
    print(f"    NO_EDGE={zero_robustness_counts['NO_EDGE']}")

    # Full-universe lookback sweep with realistic costs
    print("\n  === Realistic costs ===")
    realistic_lookback_results = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb, cfg_realistic) for t in ticker_order}
                                  for lb in LOOKBACKS}

    realistic_per_asset = {}
    realistic_robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        profile = {str(lb): realistic_lookback_results[lb][ticker] for lb in LOOKBACKS}
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        realistic_robustness_counts[v] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        realistic_per_asset[ticker] = profile

    print("  lookback-robustness verdict counts (realistic costs):")
    print(f"    ROBUST={realistic_robustness_counts['ROBUST']} (>=2 lookbacks positive edge)")
    print(f"    SENSITIVE={realistic_robustness_counts['SENSITIVE']} (edge at exactly one lookback)")
    print(f"    NO_EDGE={realistic_robustness_counts['NO_EDGE']}")

    # Summary comparison
    print("\n=== 5. Cost-sensitivity summary ===")
    print("  Comparing zero cost vs realistic costs:")
    print(f"    ROBUST assets unchanged: {realistic_robustness_counts['ROBUST'] == zero_robustness_counts['ROBUST']}")
    print(f"    SENSITIVE assets unchanged: {realistic_robustness_counts['SENSITIVE'] == zero_robustness_counts['SENSITIVE']}")
    print(f"    NO_EDGE assets unchanged: {realistic_robustness_counts['NO_EDGE'] == zero_robustness_counts['NO_EDGE']}")

    # Calculate average edge reduction due to costs
    zero_edges = sum(1 for ticker in ticker_order if zero_per_asset[ticker]["robustness_verdict"] == "ROBUST")
    realistic_edges = sum(1 for ticker in ticker_order if realistic_per_asset[ticker]["robustness_verdict"] == "ROBUST")
    edge_preservation_rate = realistic_edges / zero_edges if zero_edges > 0 else 0

    print(f"    ROBUST assets: {zero_edges} -> {realistic_edges} ({edge_preservation_rate:.1%})")

    # Determinism check for zero cost (same as original momentum_lookback_sweep)
    print("\n=== 6. Determinism check (zero cost) ===")
    res2 = {lb: {t: segment_result_with_costs(t, tickers[t], lb, cfg_zero) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: zero_lookback_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write artifact
    # Create artifact with simple structure to avoid serialization issues
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        zero_cost_robustness_counts=dict(zero_robustness_counts),
        realistic_cost_robustness_counts=dict(realistic_robustness_counts),
        zero_cost_robust_assets=[t for t in ticker_order if zero_per_asset[t]["robustness_verdict"] == "ROBUST"],
        realistic_cost_robust_assets=[t for t in ticker_order if realistic_per_asset[t]["robustness_verdict"] == "ROBUST"],
        cost_model=dict(
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("\nNOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost sensitivity check (lookback 3/5/10): "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
