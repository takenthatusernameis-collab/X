"""Real-data transaction-cost sensitivity for short-horizon momentum.

Objective: classify whether the candidate short-horizon momentum edge
(admitted as candidate positive evidence in activations 37318950814 and
37361000967) survives conservative transaction-cost stress on the collected
universe. This is the decision-relevant qualification left open by the
momentum lookback sweep.

The design is fixed a-priori and is NOT tuned to the cost results:

- Signal: long the previous ``lookback``-day return, hold 1 day, daily
  rebalance (``research.backtest.momentum_signals``), lookback in {3, 5, 10}.
- Universe: the 10 collected adjusted-close tickers (unchanged).
- Regime gate: the same 4 contiguous past-only volatility blocks and the
  same walk-forward windows (train=252d / test=84d / warmup=60d /
  overlap=60d, min 400 bars/segment) as ``momentum_lookback_sweep.py``.
- Matched null: ``stress_segments`` runs the coin-flip null through the
  framework's ``noise_benchmark`` with the SAME ``BacktestConfig`` (same
  costs) as the candidate, so the candidate-vs-null comparison is
  cost-matched at every cost level.
- Cost grid (predeclared, bounded, 3 levels):
    zero         : no commission, no slippage (reference; must reproduce the
                   prior lookback sweep exactly).
    realistic    : the repository's existing realistic cost model, copied
                   verbatim from examples/ma_crossover_real_data.py:
                   commission_per_trade=2.0, commission_per_share=0.003,
                   slippage_cents=2.0, slippage_proportional=0.0005.
    conservative : 2x the realistic model (a conservative stress level).

Cost-robustness classification (per asset, fixed rule):
- An asset shows an EDGE at a cost level if >= 2 of the 3 lookbacks are
  REGIME_STABLE with uniformly positive segment medians.
- NO_EDGE        : no EDGE even at zero cost.
- COST_ROBUST    : EDGE at zero cost and still an EDGE under conservative costs.
- COST_ERODED    : EDGE at zero cost and under realistic costs, but not under
                   conservative costs.
- COST_ELIMINATED: EDGE at zero cost but no EDGE under realistic costs.

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

# Predeclared, bounded cost grid. The 'realistic' entry is the repository's
# canonical realistic cost model (examples/ma_crossover_real_data.py); the
# 'conservative' entry is exactly 2x it.
COST_LEVELS = [
    {
        "name": "zero",
        "commission_per_trade": 0.0,
        "commission_per_share": 0.0,
        "slippage_cents": 0.0,
        "slippage_proportional": 0.0,
    },
    {
        "name": "realistic",
        "commission_per_trade": 2.0,
        "commission_per_share": 0.003,
        "slippage_cents": 2.0,
        "slippage_proportional": 0.0005,
    },
    {
        "name": "conservative",
        "commission_per_trade": 4.0,
        "commission_per_share": 0.006,
        "slippage_cents": 4.0,
        "slippage_proportional": 0.001,
    },
]


def cfg_from_cost(cost):
    """Build the BacktestConfig for one cost level (signal sizing unchanged)."""
    return bt.BacktestConfig(
        commission_per_trade=float(cost["commission_per_trade"]),
        commission_per_share=float(cost["commission_per_share"]),
        slippage_cents=float(cost["slippage_cents"]),
        slippage_proportional=float(cost["slippage_proportional"]),
    )


def is_edge(entry):
    """True if one (cost, lookback, ticker) result is a positive REGIME_STABLE edge."""
    return entry["verdict"] == "REGIME_STABLE" and all(m > 0 for m in entry["medians"])


def turnover_profile(bars, lookback, cfg):
    """Full-sample signal composition and cost totals for one ticker/cost level.

    This records WHY the cost model does or does not bite: the fraction of
    bars long/short/neutral, the number of position flips, the total traded
    notional, the total commission, and the net cash impact of slippage. A
    nearly-always-long signal rebalances a constant notional and has low
    effective turnover; an alternating long/short signal would trade far more.
    """
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=lookback)
    weights = np.array([s.weight for s in sig], dtype=np.float64)
    res = bt.run_bars(list(bars), sig, cfg)
    traded_notional = float(sum(
        abs(f.shares) * closes[f.date - 1] for f in res.trades))
    commission = float(sum(f.commission for f in res.trades))
    slippage_impact = float(sum(
        -f.shares * (f.price - closes[f.date - 1]) for f in res.trades))
    return dict(
        n_bars=int(len(closes)),
        n_long=int(np.sum(weights > 0)),
        n_short=int(np.sum(weights < 0)),
        n_neutral=int(np.sum(weights == 0)),
        n_flips=int(np.sum(np.diff(weights) != 0)),
        traded_notional=round(traded_notional, 1),
        commission=round(commission, 1),
        slippage_net_cash_impact=round(slippage_impact, 1),
        total_cost=round(commission + slippage_impact, 1),
        final_equity=round(float(res.equity_curve[-1]), 2),
    )


def segment_result(bars, lookback, cfg):
    """One ticker/lookback/cost walk-forward regime-stability result."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"signals {len(sig)} != bars {len(closes)}")
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
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(float(s.baseline_median_log_return), 3) for s in res.scenarios],
        null_medians=[round(float(s.noise_median_log_return), 3) for s in res.scenarios],
        candidate_dispersion=round(float(res.candidate_dispersion), 3),
        null_dispersion=round(float(res.null_dispersion), 3),
        n_folds=int(res.n_folds),
        verdict=res.overall_verdict,
    )


def classify(per_ticker_edges):
    """Apply the predeclared per-asset cost-robustness rule."""
    out = {}
    for ticker in per_ticker_edges:
        edges = per_ticker_edges[ticker]
        zero_robust = len(edges["zero"]) >= 2
        real_robust = len(edges["realistic"]) >= 2
        cons_robust = len(edges["conservative"]) >= 2
        if not zero_robust:
            cls = "NO_EDGE"
        elif cons_robust:
            cls = "COST_ROBUST"
        elif real_robust:
            cls = "COST_ERODED"
        else:
            cls = "COST_ELIMINATED"
        out[ticker] = dict(
            zero_edge_lookbacks=edges["zero"],
            realistic_edge_lookbacks=edges["realistic"],
            conservative_edge_lookbacks=edges["conservative"],
            zero_robust=zero_robust,
            realistic_robust=real_robust,
            conservative_robust=cons_robust,
            cost_robustness=cls,
        )
    return out


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
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

    print("\n=== 3. Leakage review per asset (realistic cost model) ===")
    realistic = next(c for c in COST_LEVELS if c["name"] == "realistic")
    cfg_real = cfg_from_cost(realistic)
    tickers = {}
    for ticker in ["AMZN", "JPM"]:
        bars, _dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cfg_real)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        n_trades = len(res.trades)
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], fills {n_trades}, total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "costs apply to every fill (commission + slippage).")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    print("\n=== 4. Momentum cost-sensitivity grid (lookback 3/5/10 x 3 cost levels) ===")
    for cost in COST_LEVELS:
        print("  cost {:12s}: commission_per_trade={:.2f} commission_per_share={:.4f} "
              "slippage_cents={:.2f} slippage_proportional={:.5f}".format(
                  cost["name"], cost["commission_per_trade"],
                  cost["commission_per_share"], cost["slippage_cents"],
                  cost["slippage_proportional"]))
    print("")

    results = {}
    for cost in COST_LEVELS:
        cfg = cfg_from_cost(cost)
        name = cost["name"]
        results[name] = {}
        for lb in LOOKBACKS:
            results[name][str(lb)] = {}
            for ticker in ticker_order:
                results[name][str(lb)][ticker] = segment_result(
                    all_tickers[ticker], lb, cfg)
        print("  cost {:12s} done".format(name))

    # Edge flags per (ticker, cost level).
    per_ticker_edges = {}
    for ticker in ticker_order:
        per_ticker_edges[ticker] = {}
        for cost in COST_LEVELS:
            name = cost["name"]
            per_ticker_edges[ticker][name] = [
                lb for lb in LOOKBACKS
                if is_edge(results[name][str(lb)][ticker])
            ]

    classification = classify(per_ticker_edges)

    print("\n=== 4b. Turnover / signal composition (lookback 5, per cost level) ===")
    print("  ticker  cost          long  short  neutral  flips  traded_notional  commission  slippage_net  total_cost")
    turnover = {}
    for cost in COST_LEVELS:
        name = cost["name"]
        turnover[name] = {}
        for ticker in ticker_order:
            tp = turnover_profile(all_tickers[ticker], 5, cfg_from_cost(cost))
            turnover[name][ticker] = tp
            print("  {:6s} {:12s} {:5d} {:6d} {:8d} {:6d} {:16.1f} {:11.1f} {:12.1f} {:10.1f}".format(
                ticker, name, tp["n_long"], tp["n_short"], tp["n_neutral"],
                tp["n_flips"], tp["traded_notional"], tp["commission"],
                tp["slippage_net_cash_impact"], tp["total_cost"]))
        print("")

    edge_cells_by_cost = {}
    verdict_counts_by_cost = {}
    for cost in COST_LEVELS:
        name = cost["name"]
        edge_cells_by_cost[name] = {}
        verdict_counts_by_cost[name] = {}
        for lb in LOOKBACKS:
            edge_cells_by_cost[name][str(lb)] = sum(
                1 for t in ticker_order if is_edge(results[name][str(lb)][t]))
            counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                      "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
            for t in ticker_order:
                counts[results[name][str(lb)][t]["verdict"]] += 1
            verdict_counts_by_cost[name][str(lb)] = counts

    robustness_counts = {"COST_ROBUST": 0, "COST_ERODED": 0,
                         "COST_ELIMINATED": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        robustness_counts[classification[ticker]["cost_robustness"]] += 1

    n_zero_robust = sum(1 for t in ticker_order if classification[t]["zero_robust"])
    n_cons_robust = sum(1 for t in ticker_order if classification[t]["conservative_robust"])

    print("\n=== 5. Per-asset cost-robustness classification ===")
    for ticker in ticker_order:
        c = classification[ticker]
        print("  {:6s} zero={:<10s} realistic={:<10s} conservative={:<10s} -> {}".format(
            ticker,
            ",".join(str(x) for x in c["zero_edge_lookbacks"]) or "-",
            ",".join(str(x) for x in c["realistic_edge_lookbacks"]) or "-",
            ",".join(str(x) for x in c["conservative_edge_lookbacks"]) or "-",
            c["cost_robustness"]))
    print("")
    print("  cost-robustness counts: {}".format(robustness_counts))
    print("  assets with a robust positive edge at zero cost:        {}/10".format(n_zero_robust))
    print("  assets with a robust positive edge under conservative:  {}/10".format(n_cons_robust))
    print("  EDGE cells per cost level (of 10 assets x 3 lookbacks = 30):")
    for cost in COST_LEVELS:
        name = cost["name"]
        cells = [edge_cells_by_cost[name][str(lb)] for lb in LOOKBACKS]
        print("    {:12s}: lookback 3/5/10 -> {} (total {})".format(
            name, cells, sum(cells)))
    print("")

    # Determinism: rerun the AMZN/JPM subset at every cost level/lookback.
    res2 = {cost["name"]: {str(lb): {t: segment_result(tickers[t], lb, cfg_from_cost(cost))
                                     for t in ["AMZN", "JPM"]}
                           for lb in LOOKBACKS}
            for cost in COST_LEVELS}
    out1 = json.dumps({c["name"]: {str(lb): {t: results[c["name"]][str(lb)][t]
                                             for t in ["AMZN", "JPM"]}
                                   for lb in LOOKBACKS}
                       for c in COST_LEVELS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_levels=[dict(c) for c in COST_LEVELS],
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        results=results,
        turnover=turnover,
        classification=classification,
        summary=dict(
            edge_cells_by_cost=edge_cells_by_cost,
            verdict_counts_by_cost=verdict_counts_by_cost,
            cost_robustness_counts=robustness_counts,
            n_tickers_with_zero_robust_edge=n_zero_robust,
            n_tickers_with_conservative_robust_edge=n_cons_robust,
            ticker_order=ticker_order,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum transaction-cost sensitivity across the "
          "collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())