"""Real-data transaction-cost sensitivity check for short-horizon momentum.

Objective (focused task R-001, campaign slot 2): the candidate positive
momentum evidence (lookback 3/5/10, `momentum_lookback_sweep.py`) was produced
at zero/low cost. Momentum is a mechanical 1-day-rebalance signal, so its edge
is decision-relevant only if it survives conservative transaction-cost stress.
This check runs a *predeclared* cost grid through the existing walk-forward
regime-stability gate with a matched coin-flip null, on the collected 10-asset
universe, and classifies cost robustness. It does not tune the signal, expand
the universe, or add leverage.

Predeclared design (fixed a-priori, NOT tuned to results):
- Lookbacks: 3, 5, 10 (the same set as the admitted momentum evidence).
- Cost levels (proportional slippage, on top of the repository's base real
  cost model of commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0):
    * zero    : all costs 0 (reference; reproduces the frictionless result)
    * base    : commission_per_trade=2.0, commission_per_share=0.003,
                slippage_cents=2.0, slippage_proportional=0.0005  (5 bps)
    * double  : base with slippage_proportional=0.0010 (10 bps)
    * quad    : base with slippage_proportional=0.0020 (20 bps)
  The grid is a monotone cost ladder; the base level is the repository's
  existing realistic cost model, and double/quad are conservative stress.
- Walk-forward: train=252d / test=84d / warmup=60d / overlap=60d, 4 contiguous
  volatility blocks, min 400 bars/segment; each (lookback, cost) cell compared
  vs a coin-flip null run on the same segments *with the same cost cfg*.

A-priori classification (recorded before execution, not observed):
- COST_ROBUST   : the momentum edge (positive REGIME_STABLE medians) survives at
                  the base cost level for the same assets that show it at zero
                  cost, and the universe edge count does not collapse.
- COST_FRAGILE  : the edge attenuates materially or disappears at the base cost
                  level relative to zero cost.
- COST_INCONCLUSIVE: mixed/ambiguous evidence across lookbacks.
The prediction is recorded as a hypothesis; the observed classification is
derived mechanically from the artifact medians below.

Materiality threshold (predeclared): a cost level is MATERIAL if it changes any
segment median by more than MATERIALITY = 0.05 in absolute log-return terms.
0.05 is the framework's own OUTLIER_TOL / fixed null band
(`research/backtest/regime_stability.py`), reused here as the predeclared
materiality scale so the classification does not flip on sub-noise verdict
changes. The discrete REGIME_STABLE/REGIME_DEPENDENT verdict can change when a
cost perturbation nudges the coin-flip null dispersion across the 2x boundary
even though no segment median moved materially; the magnitude check is the
operative robustness test and the verdict is reported alongside it.

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

# Predeclared cost ladder. `base` is the repository's existing realistic cost
# model; double/quad raise only the proportional slippage (a conservative
# stress on the dominant cost for a daily-rebalanced large-cap signal).
COST_LEVELS = {
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "base": dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005),
    "double": dict(commission_per_trade=2.0, commission_per_share=0.003,
                   slippage_cents=2.0, slippage_proportional=0.0010),
    "quad": dict(commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0020),
}
COST_ORDER = ("zero", "base", "double", "quad")

# Predeclared materiality scale (absolute log-return change in any segment
# median). Reuses the framework's OUTLIER_TOL / fixed null band (0.05).
MATERIALITY = 0.05

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cfg_for(level):
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM,
                             **COST_LEVELS[level])


def cell_result(ticker, bars, lookback, level):
    """Walk-forward regime-stability result for one ticker / lookback / cost."""
    closes = bars.closes_array()
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
        cfg=cfg_for(level),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def is_edge(cell):
    """A positive REGIME_STABLE edge: uniformly positive medians."""
    return cell["verdict"] == "REGIME_STABLE" and all(m > 0 for m in cell["medians"])


def max_abs_median_change(cells_zero, cells_level):
    """Largest absolute per-segment median change between two cost levels."""
    return max(abs(a - b)
               for za, zl in zip(cells_zero["medians"], cells_level["medians"])
               for a, b in [(za, zl)])


def classify_cost_robustness(per_asset):
    """Mechanically classify cost robustness from the artifact medians.

    Two distinct signals are computed and reported separately, because they can
    disagree:

    1. MAGNITUDE (operative): does any asset's per-segment median move by more
       than the predeclared MATERIALITY (0.05) at any cost level vs zero cost?
       This is the economically meaningful test — a real edge collapse moves
       medians materially.
    2. STRICT VERDICT: does any zero-cost EDGE asset lose its positive
       REGIME_STABLE verdict at base cost? The discrete verdict can flip when a
       cost perturbation nudges the coin-flip null dispersion across the 2x
       boundary even though no median moved materially (a null-dispersion
       artifact, not an edge collapse).

    Classification:
    - COST_ROBUST   : no material median change AND no zero-cost edge lost.
    - COST_FRAGILE  : material median change at some cost level (a genuine
                      cost sensitivity), OR (no material change but a zero-cost
                      edge lost) -> reported as COST_FRAGILE_VERDICT_ONLY so the
                      knife-edge nature is explicit and not mistaken for a real
                      collapse.
    - COST_INCONCLUSIVE: no zero-cost edges to qualify.
    """
    zero_edges = [t for t in per_asset if is_edge(per_asset[t]["5"]["zero"])]
    base_edges = [t for t in per_asset if is_edge(per_asset[t]["5"]["base"])]
    lost = [t for t in zero_edges if t not in base_edges]

    material = {}
    for t in per_asset:
        worst = 0.0
        for lb in per_asset[t]:
            z = per_asset[t][lb]["zero"]["medians"]
            for lvl in COST_ORDER:
                if lvl == "zero":
                    continue
                lev = per_asset[t][lb][lvl]["medians"]
                worst = max(worst, max(abs(a - b) for a, b in zip(z, lev)))
        material[t] = round(worst, 4)
    material_assets = [t for t, w in material.items() if w > MATERIALITY]

    detail = dict(
        zero_edges=zero_edges, base_edges=base_edges, lost=lost,
        material_assets=material_assets, max_median_change=material,
        strict_verdict_fragile=bool(lost))

    if not zero_edges:
        return "COST_INCONCLUSIVE", detail
    if material_assets:
        return "COST_FRAGILE", detail
    if lost:
        return "COST_FRAGILE_VERDICT_ONLY", detail
    return "COST_ROBUST", detail


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

    print("\n=== 3. Leakage review per asset (lookback=5, zero cost) ===")
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

    print("\n=== 4. Cost-sensitivity grid (lookback x cost) across collected universe ===")
    print("  Predeclared cost ladder (proportional slippage on top of base "
          "commission/slippage):")
    for lvl in COST_ORDER:
        c = COST_LEVELS[lvl]
        print(f"    {lvl:7s}: comm/trade={c['commission_per_trade']}, "
              f"comm/share={c['commission_per_share']}, "
              f"slip_cents={c['slippage_cents']}, "
              f"slip_prop={c['slippage_proportional']}")
    print("  Walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 volatility blocks, each cell vs "
          "a coin-flip null on the same segments with the same cost cfg\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][cost][ticker].
    results = {lb: {lvl: {t: cell_result(t, all_tickers[t], lb, lvl)
                          for t in ticker_order}
                    for lvl in COST_ORDER}
               for lb in LOOKBACKS}

    # Edge counts per (lookback, cost) and per-asset robustness classification.
    edge_counts = {}
    for lb in LOOKBACKS:
        for lvl in COST_ORDER:
            edge_counts[f"{lb}/{lvl}"] = sum(
                1 for t in ticker_order if is_edge(results[lb][lvl][t]))

    per_asset = {}
    for t in ticker_order:
        per_asset[t] = {
            str(lb): {lvl: results[lb][lvl][t] for lvl in COST_ORDER}
            for lb in LOOKBACKS
        }
    classification, class_detail = classify_cost_robustness(per_asset)

    print("  Edge counts (positive REGIME_STABLE) per lookback/cost:")
    for lb in LOOKBACKS:
        row = "  lookback {}: ".format(lb) + "  ".join(
            "{}={}".format(lvl, edge_counts[f"{lb}/{lvl}"]) for lvl in COST_ORDER)
        print(row)
    print("")

    # Determinism: rerun the AMZN/JPM subset at lookback 5, base cost.
    res2 = {t: cell_result(t, tickers[t], 5, "base") for t in target}
    out1 = json.dumps({t: results[5]["base"][t] for t in target}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_order=list(COST_ORDER),
        cost_levels=COST_LEVELS,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        edge_counts=edge_counts,
        classification=classification,
        classification_detail=class_detail,
        per_asset=per_asset,
        universe=dict(
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): {
                        lvl: per_asset[t][str(lb)][lvl]["medians"]
                        for lvl in COST_ORDER}
                        for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): {
                        lvl: per_asset[t][str(lb)][lvl]["null_medians"]
                        for lvl in COST_ORDER}
                        for lb in LOOKBACKS},
                    verdicts={str(lb): {
                        lvl: per_asset[t][str(lb)][lvl]["verdict"]
                        for lvl in COST_ORDER}
                        for lb in LOOKBACKS},
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("classification (lookback 5, zero->base): {} {}".format(
        classification, class_detail))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())