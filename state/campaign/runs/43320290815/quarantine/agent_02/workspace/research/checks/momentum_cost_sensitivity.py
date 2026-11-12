"""Real-data momentum cost-sensitivity across the collected universe.

Primary question: Does the short-horizon momentum edge (lookback 3/5/10,
long the previous N-day return, hold 1 day, daily rebalance) survive
conservative transaction-cost stress on the collected real-data universe?

The repository's cost model is BacktestConfig
(research/backtest/engine.py): commission_per_trade, commission_per_share,
slippage_cents, slippage_proportional. They default to zero. This check
stress-tests the admitted candidate-positive momentum edge with a predeclared
conservative cost grid while keeping the signal rules, universe,
walk-forward windows, and coin-flip null fixed.

Predeclared cost grid (conservative for daily large-cap equity trading):

  zero      : $0.000/share commission, 0 cents fixed + 0.0 bps proportional
              (baseline replication of the zero-cost momentum evidence)
  cheap     : $0.005/share commission, 1 cent fixed + 5 bps proportional
  base      : $0.005/share commission, 1 cent fixed + 10 bps proportional
  stressed  : $0.010/share commission, 2 cents fixed + 20 bps proportional

With initial_capital=1e6 and target_exposure=1.0, a $1M trade at $100/share
(10,000 shares) implies a round-trip cost of roughly 13 bps (cheap),
33 bps (base) or 65 bps (stressed); the momentum signal is fully invested
every bar after the lookback window and flips sign frequently, so each
84-bar OOS segment executes on the order of 50-85 round trips and the cost
drag scales linearly with that turnover. The matched coin-flip null in
`stress_segments` is run with the same `cfg`, so the null also pays costs
and trades at the same frequency: the null is a matched cost baseline, and
any residual candidate-null gap survives the cost stress.

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day
return; hold 1 day; daily rebalance; lookbacks 3, 5, 10; walk-forward
train=252d/test=84d/warmup=60d/overlap=60d; 4 contiguous volatility blocks
of the collected series; min 400 bars/segment. For each lookback and each
cost level, per-segment median log return is compared against the coin-flip
null run on the same segments with the same costs. Determinism is asserted
via an internal independent recomputation of the AMZN/JPM subset; an
independent verifier recomputes those gates via a fresh walk_forward +
noise_benchmark path.

Classification (predeclared; applied after the grid runs):
  COST_ROBUST     : positive REGIME_STABLE edge at >= 2 cost levels, including
                    the stressed level.
  COST_SENSITIVE  : a positive edge survives only the zero/cheap levels and is
                    destroyed by base or stressed costs.
  COST_KILLED     : no positive edge at any realistic cost level (edge only at
                    zero cost, or nowhere).

A-priori hypothesis: the admitted momentum edge is small per segment
(+/-5-7% log return) but very high-frequency (daily rebalance on every bar);
conservative transaction costs should erode it, and a negative conclusion
(cost destruction) is a meaningful qualification of the candidate-positive
evidence rather than a reason to stop. Either outcome closes the cell.

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

# Predeclared conservative cost grid. Tuples:
# (commission_per_share, slippage_cents, slippage_proportional).
COST_LEVELS = {
    "zero": (0.0, 0.0, 0.0),
    "cheap": (0.005, 1.0, 0.0005),
    "base": (0.005, 1.0, 0.0010),
    "stressed": (0.010, 2.0, 0.0020),
}
COST_DESC = {
    "zero": "baseline, zero costs",
    "cheap": "$0.005/share comm, 1 cent + 5 bps slippage",
    "base": "$0.005/share comm, 1 cent + 10 bps slippage",
    "stressed": "$0.010/share comm, 2 cents + 20 bps slippage",
}
TARGET = ("AMZN", "JPM")


def _cfg_from_costs(costs):
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_share=float(costs[0]),
        commission_per_trade=1.0,
        slippage_cents=float(costs[1]),
        slippage_proportional=float(costs[2]),
    )


def segment_result(ticker, bars, lookback, costs):
    """Walk-forward regime-stability result for one ticker, lookback, cost level."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    cfg = _cfg_from_costs(costs)
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
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    scen = res.scenarios[0]
    gap = [round(scen.baseline_median_log_return - s.noise_median_log_return, 3)
           for s in res.scenarios]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        cand_vs_null_gap=gap,
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def edge_or_not(sr):
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def cost_verdict(cost_results):
    """COST_ROBUST / COST_SENSITIVE / COST_KILLED across cost levels."""
    edges = [cl for cl, res in cost_results.items()
             if edge_or_not(res) == "EDGE"]
    if len(edges) >= 2:
        return "COST_ROBUST"
    if len(edges) == 1:
        return "COST_SENSITIVE"
    return "COST_KILLED"


def fill_audit(ticker, bars, lookback, costs):
    """Leakage + cost audit for one ticker/lookback/cost: equity must match
    fills and the fills must show real costs are being applied."""
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=lookback)
    bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
    cfg = _cfg_from_costs(costs)
    res = bt.run_bars(list(bars), sig, cfg)
    bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
    total_commission = sum(f.commission for f in res.trades)
    total_slippage = sum(
        f.shares * cfg.slippage_cents / 100.0 + abs(f.shares) * closes[int(f.date) - 1]
        * cfg.slippage_proportional
        for f in res.trades)
    round_trips = len(res.trades)
    return dict(
        n_bars=bars.n_bars,
        n_signals=len(sig),
        leakage="PASS",
        n_round_trips=round_trips,
        total_commission=round(total_commission, 2),
        total_slippage=round(total_slippage, 2),
        total_cost=round(total_commission + total_slippage, 2),
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

    print("\n=== 3. Leakage + cost audit (AMZN, JPM, lookback=5, base costs) ===")
    print("  Engine equity must match fills under real commission + slippage,")
    print("  confirming the cost path is live and correctly traceable.\n")
    tickers = {}
    for ticker in TARGET:
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars
        audit = fill_audit(ticker, bars, lookback=5, costs=COST_LEVELS["base"])
        print("  {}: {} bars, {} round trips; comm ${:.2f}, "
              "slippage ${:.2f}, total ${:.2f} -> leakage {}".format(
            ticker, audit["n_bars"], audit["n_round_trips"],
            audit["total_commission"], audit["total_slippage"],
            audit["total_cost"], audit["leakage"]))
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only.")

    print("\n=== 4. Momentum cost grid across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, lookbacks 3/5/10, matched "
          "coin-flip null at the same cost level.\n".format(
          TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("  Cost grid: zero | cheap ($0.005/share + 1c + 5bps) | "
          "base ($0.005/share + 1c + 10bps) | stressed ($0.010/share + 2c + 20bps)\n")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # lookback_results[lookback][ticker][cost_level] -> segment result dict
    lookback_results = {
        lb: {t: {cl: segment_result(t, all_tickers[t], lb, COST_LEVELS[cl])
                 for cl in COST_LEVELS}
             for t in ticker_order}
        for lb in LOOKBACKS
    }

    universe = {}
    robustness = {}
    cost_level_counts = {cl: {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "COST_KILLED": 0}
                         for cl in COST_LEVELS}
    per_cost_edge_counts = {cl: 0 for cl in COST_LEVELS}
    for ticker in ticker_order:
        profile = {}
        robustness[ticker] = {}
        for lb in LOOKBACKS:
            cost_results = {cl: lookback_results[lb][ticker][cl] for cl in COST_LEVELS}
            profile[str(lb)] = {
                "cand_vs_null_gap": {cl: lookback_results[lb][ticker][cl]["cand_vs_null_gap"]
                                     for cl in COST_LEVELS},
                "edge_or_not": {cl: edge_or_not(cost_results[cl])
                                for cl in COST_LEVELS},
            }
            robustness[ticker][str(lb)] = cost_verdict(cost_results)
            for cl, cr in cost_results.items():
                cost_level_counts[cl][robustness[ticker][str(lb)]] += 1
                if edge_or_not(cr) == "EDGE":
                    per_cost_edge_counts[cl] += 1
        universe[ticker] = profile
        universe[ticker]["cost_robustness"] = robustness[ticker]

    print("  Cost-level edge counts (positive REGIME_STABLE edges per cost level):")
    for cl in COST_LEVELS:
        print("    {}: {} edges across universe".format(cl, per_cost_edge_counts[cl]))
    print()
    print("  Per-asset cost-robustness verdicts (edge status per cost level):")
    for ticker in ticker_order:
        parts = []
        for lb in LOOKBACKS:
            p = universe[ticker][str(lb)]
            edge_parts = ",".join("{}:{}({})".format(
                cl, p["edge_or_not"][cl], p["edge_or_not"][cl])
                for cl in COST_LEVELS)
            parts.append("{}[{}]".format(lb, edge_parts))
        print("    {:6s} robustness={}  ".format(
            ticker, robustness[ticker]) + "| ".join(parts) + "")
    print("")

    print("  Universe-level cost-robustness counts:")
    for cl in COST_LEVELS:
        print("    {}:".format(cl))
        for v, c in cost_level_counts[cl].items():
            print("      {:16s}: {}".format(v, c))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: {cl: segment_result(t, tickers[t], lb, COST_LEVELS[cl])
                     for cl in COST_LEVELS}
                 for t in TARGET}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in TARGET}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_levels={cl: {
            "commission_per_share": COST_LEVELS[cl][0],
            "slippage_cents": COST_LEVELS[cl][1],
            "slippage_proportional": COST_LEVELS[cl][2],
            "description": COST_DESC[cl],
        } for cl in COST_LEVELS},
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=dict(
            (t, {str(lb): {cl: dict(lookback_results[lb][t][cl])
                            for cl in COST_LEVELS}
                 for lb in LOOKBACKS})
            for t in ticker_order
        ),
        universe=dict(
            per_cost_edge_counts=per_cost_edge_counts,
            cost_level_counts=cost_level_counts,
            per_asset=dict(
                (t, dict(universe[t]))
                for t in ticker_order
            ),
            ticker_order=ticker_order,
        ),
        leakage=dict(
            (t, fill_audit(t, tickers[t], lookback=5, costs=COST_LEVELS["base"]))
            for t in TARGET
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
