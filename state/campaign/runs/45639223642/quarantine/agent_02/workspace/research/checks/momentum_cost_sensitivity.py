"""Real-data transaction-cost sensitivity of short-horizon momentum.

Objective (task R-001): decide whether the short-horizon momentum edge admitted
as candidate positive evidence (`research/checks/momentum.py`, activation
37318950814; lookback sweep 37361000967) survives conservative transaction-cost
stress on the collected 10-asset adjusted-close universe, under the repository's
existing realistic cost model and a matched coin-flip null.

Predeclared cost grid (fixed a-priori, NOT tuned to results):
  zero         : no commission / no slippage (reference).
  realistic    : the repository's existing realistic model
                 (commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005), matching
                 examples/ma_crossover_real_data.py.
  conservative : 2x the realistic variable costs
                 (commission_per_trade=4.0, commission_per_share=0.006,
                 slippage_cents=4.0, slippage_proportional=0.0010).

Lookbacks {3, 5, 10} (the predeclared short-horizon set). For each ticker,
lookback, and cost level, momentum (long the previous N-day return, hold 1 day,
daily rebalance) is walk-forward validated (train=252d/test=84d/warmup=60d/
overlap=60d) inside each of the 4 contiguous volatility blocks, and each
segment's candidate median log return is compared against a coin-flip sign null
run on the SAME segments with the SAME cost cfg (a cost-matched null). The
synthetic generator cannot answer this (no momentum structure), so the null is
drawn from the collected data itself.

Predeclared classification rule (fixed before execution):
  zero_cost_edge(asset, lb)      : zero-cost verdict is REGIME_STABLE with all
                                   segment medians > 0.
  survives_conservative(asset,lb): conservative-cost verdict is REGIME_STABLE
                                   with all segment medians > 0.
  Per-asset cost-robustness:
    ROBUST   : >= 2 lookbacks with a zero-cost edge AND all of those also
               survive conservative cost.
    FRAGILE  : >= 1 zero-cost edge but NONE survive conservative cost.
    NO_EDGE  : no zero-cost edge at any lookback.
  Overall: COST_ROBUST if a majority of the zero-cost-edge assets are ROBUST;
  COST_FRAGILE if a majority are FRAGILE; MIXED otherwise.

A-priori prediction (recorded before results): the repo's constant-notional
engine charges costs on actual traded notional, and the per-fold momentum edge
(~0.05-0.13 log return) is large relative to the per-fold cost at the realistic
and conservative levels, so the edge is expected to survive conservative stress
(COST_ROBUST), with only mild median compression. If instead conservative costs
erase the edge for most assets, the momentum admission must be qualified as
cost-fragile.

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

# Predeclared cost grid (a-priori; see module docstring).
COST_LEVELS = {
    "zero": dict(),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010),
}
COST_ORDER = ("zero", "realistic", "conservative")

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def segment_result(ticker, bars, lookback, cost):
    """Walk-forward regime-stability result for one ticker/lookback/cost level."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    cfg = bt.BacktestConfig(initial_capital=1e6, **COST_LEVELS[cost])
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
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 6) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 6) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 6),
        null_dispersion=round(res.null_dispersion, 6),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def is_positive_edge(sr):
    """REGIME_STABLE with uniformly positive medians = a positive stable edge."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def classify_asset(per_lb_cost):
    """Per-asset cost-robustness verdict from {lb: {cost: segment_result}}."""
    zero_edges = [lb for lb in LOOKBACKS if is_positive_edge(per_lb_cost[lb]["zero"])]
    survivors = [lb for lb in zero_edges
                 if is_positive_edge(per_lb_cost[lb]["conservative"])]
    if len(zero_edges) >= 2 and len(survivors) == len(zero_edges):
        return "ROBUST", zero_edges, survivors
    if len(zero_edges) >= 1 and len(survivors) == 0:
        return "FRAGILE", zero_edges, survivors
    if len(zero_edges) == 0:
        return "NO_EDGE", zero_edges, survivors
    # Some but not all zero-cost edges survive conservative cost.
    return "PARTIAL", zero_edges, survivors


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

    print("\n=== 4. Momentum cost-sensitivity sweep (lookback 3/5/10 x cost "
          "zero/realistic/conservative) ===")
    print("  Predeclared cost grid (a-priori):")
    for name in COST_ORDER:
        print(f"    {name:12s}: {COST_LEVELS[name]}")
    print("  Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 contiguous volatility blocks, "
          "cost-matched coin-flip null on the same segments\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # results[ticker][lb][cost] = segment_result
    results = {t: {lb: {c: segment_result(t, all_tickers[t], lb, c) for c in COST_ORDER}
                   for lb in LOOKBACKS}
               for t in ticker_order}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "PARTIAL": 0, "FRAGILE": 0, "NO_EDGE": 0}
    zero_edge_asset_count = 0
    for ticker in ticker_order:
        per_lb_cost = results[ticker]
        verdict, zero_edges, survivors = classify_asset(per_lb_cost)
        robustness_counts[verdict] += 1
        if zero_edges:
            zero_edge_asset_count += 1
        # Erosion magnitude: mean per-segment median at zero vs conservative,
        # for lookbacks with a zero-cost edge (or lookback 5 if none).
        ref_lb = zero_edges[0] if zero_edges else 5
        z = per_lb_cost[ref_lb]["zero"]["medians"]
        c = per_lb_cost[ref_lb]["conservative"]["medians"]
        per_asset[ticker] = dict(
            cost_robustness=verdict,
            zero_cost_edge_lookbacks=zero_edges,
            conservative_survivor_lookbacks=survivors,
            reference_lookback=ref_lb,
            zero_medians_ref=z,
            conservative_medians_ref=c,
            erosion_ref=round(float(np.mean(c) - np.mean(z)), 6),
            per_lookback={
                str(lb): {
                    "zero": per_lb_cost[lb]["zero"],
                    "realistic": per_lb_cost[lb]["realistic"],
                    "conservative": per_lb_cost[lb]["conservative"],
                } for lb in LOOKBACKS
            },
        )

    print("  Per-asset cost-robustness:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        print("    {:6s} {:8s} zero_edges={} survivors={} erosion_ref={:+.4f}".format(
            ticker, p["cost_robustness"], p["zero_cost_edge_lookbacks"],
            p["conservative_survivor_lookbacks"], p["erosion_ref"]))
    print("")
    print("  Robustness counts: ROBUST={} PARTIAL={} FRAGILE={} NO_EDGE={}".format(
        robustness_counts["ROBUST"], robustness_counts["PARTIAL"],
        robustness_counts["FRAGILE"], robustness_counts["NO_EDGE"]))
    print("  Assets with a zero-cost positive edge at >=1 lookback: {}/10".format(
        zero_edge_asset_count))

    # Overall classification (predeclared rule).
    if zero_edge_asset_count == 0:
        overall = "NO_EDGE"
    else:
        robust_n = robustness_counts["ROBUST"]
        fragile_n = robustness_counts["FRAGILE"]
        if robust_n > zero_edge_asset_count / 2.0:
            overall = "COST_ROBUST"
        elif fragile_n > zero_edge_asset_count / 2.0:
            overall = "COST_FRAGILE"
        else:
            overall = "MIXED"
    print("\n  Overall cost-robustness classification: {}".format(overall))

    # Determinism: rerun the target subset across all lookbacks/costs.
    res2 = {t: {lb: {c: segment_result(t, tickers[t], lb, c) for c in COST_ORDER}
                for lb in LOOKBACKS}
            for t in target}
    out1 = json.dumps({t: results[t] for t in target}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_levels={k: COST_LEVELS[k] for k in COST_ORDER},
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        overall_cost_robustness=overall,
        robustness_counts=robustness_counts,
        zero_edge_asset_count=zero_edge_asset_count,
        ticker_order=ticker_order,
        per_asset=per_asset,
        a_priori_prediction=(
            "COST_ROBUST expected: repo constant-notional engine charges costs on "
            "actual traded notional; per-fold momentum edge is large relative to "
            "realistic/conservative per-fold cost, so the edge should survive "
            "conservative stress with mild median compression."
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())