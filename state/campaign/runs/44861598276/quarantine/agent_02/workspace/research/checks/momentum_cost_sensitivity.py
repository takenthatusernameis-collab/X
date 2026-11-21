"""Real-data transaction-cost sensitivity of the momentum candidate.

Campaign task R-001 (HIGHER_ORDER_OBJECTIVE, campaign slot 02).

Background: the short-horizon momentum candidate (long the previous N-day
return, hold 1 day, daily rebalance) was admitted as candidate positive
evidence at lookback 5 (`research/checks/momentum.py`, activation 37318950814)
and shown to be lookback-robust in magnitude across lookbacks 3/5/10
(`research/checks/momentum_lookback_sweep.py`, activation 37361000967). The
open qualification recorded in `state/LEARNING_STATE.md` is execution-cost
robustness: the candidate is a mechanical, daily-rebalanced, direction-only
rule, so its per-bar turnover is high and transaction costs are the most
decision-relevant remaining threat to the positive result.

Objective (bounded, predeclared): run lookback 3/5/10 momentum on the existing
10-asset collected universe through the existing regime-stability gate under a
fixed, a-priori cost grid, each cost level compared against a matched coin-flip
null run under the same cost model. Classify cost robustness per asset and in
aggregate. No signal rule is tuned after seeing cost results, the universe is
not expanded, and no leverage is introduced.

Predeclared bounded cost grid (fixed before execution):
  - zero       : the repository's default BacktestConfig (no costs).
  - realistic  : the repository's existing realistic cost model
                 ($2/trade + $0.003/share, 2c fixed + 5 bps proportional
                 slippage; see examples/ma_crossover_real_data.py,
                 examples/ma_crossover.py, examples/volatility_regime_filter.py).
  - stress     : 2x the realistic model (a conservative cost-stress level,
                 fixed a-priori, not tuned to any observed result).

Cost-robustness classification (predeclared, per asset, at lookback 5, the
admitted candidate lookback):
  - a "positive edge" is a REGIME_STABLE verdict whose segment medians are all
    > 0 (REGIME_STABLE alone can describe stability of losses; medians decide).
  - NO_ZERO_COST_EDGE  : no positive edge even at zero cost.
  - COST_ROBUST        : positive edge at realistic AND at stress.
  - COST_FRAGILE_STRESS: positive edge at realistic but lost at 2x stress.
  - COST_ELIMINATED    : positive edge at zero cost but lost at realistic costs.

Aggregate class over the universe (lookback 5, n0/nr/ns = #assets with a
positive edge at zero/realistic/stress):
  - NO_ZERO_COST_EDGE if n0 == 0; COST_ROBUST if nr == n0 and ns == n0;
    COST_FRAGILE_STRESS if nr == n0 and ns < n0; else COST_ELIMINATED.

A zero-cost cross-consistency gate is enforced by the independent verifier
against `state/check_artifacts/momentum_lookback_sweep_results.json`.

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
CANONICAL_LOOKBACK = 5

COST_LEVELS = ("zero", "realistic", "stress")

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

VERDICT_KEYS = (
    "CONSISTENT_WITH_NOISE",
    "REGIME_STABLE",
    "REGIME_STABLE_LOSS",
    "REGIME_DEPENDENT",
)


def cost_config(level):
    """Return the predeclared BacktestConfig for one cost level."""
    base = dict(initial_capital=1e6, target_exposure=1.0)
    if level == "zero":
        return bt.BacktestConfig(**base)
    if level == "realistic":
        return bt.BacktestConfig(
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
            **base,
        )
    if level == "stress":
        return bt.BacktestConfig(
            commission_per_trade=4.0,
            commission_per_share=0.006,
            slippage_cents=4.0,
            slippage_proportional=0.0010,
            **base,
        )
    raise ValueError(f"unknown cost level {level}")


def cost_grid_record():
    """Serializable record of the predeclared cost grid."""
    out = {}
    for level in COST_LEVELS:
        cfg = cost_config(level)
        out[level] = dict(
            commission_per_trade=cfg.commission_per_trade,
            commission_per_share=cfg.commission_per_share,
            slippage_cents=cfg.slippage_cents,
            slippage_proportional=cfg.slippage_proportional,
            initial_capital=cfg.initial_capital,
            target_exposure=cfg.target_exposure,
        )
    return out


def segment_result(bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker and one lookback
    under a fixed cost config. Uses the existing framework gate
    (`stress_segments`) with a coin-flip null matched to the same costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"signals {len(sig)} != bars {len(closes)}")
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
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def is_positive_edge(sr):
    """A positive edge is a REGIME_STABLE verdict with all medians > 0.

    REGIME_STABLE alone can describe stability of losses, so the medians must
    be strictly positive for the verdict to count as an edge.
    """
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def all_medians_positive(sr):
    """Verdict-independent cost-robustness view: are every segment median
    strictly positive? This is immune to the matched null's dispersion, which
    changes with the cost level and can flip REGIME_STABLE/REGIME_DEPENDENT
    labels without any change in the candidate's own returns."""
    return all(m > 0 for m in sr["medians"])


def classify_cost(edge_flags):
    """Predeclared per-asset cost-robustness classification at one lookback."""
    e0 = edge_flags["zero"]
    er = edge_flags["realistic"]
    es = edge_flags["stress"]
    if not e0:
        return "NO_ZERO_COST_EDGE"
    if er and es:
        return "COST_ROBUST"
    if er and not es:
        return "COST_FRAGILE_STRESS"
    return "COST_ELIMINATED"


def classify_aggregate(edge_counts):
    """Predeclared universe-level cost-robustness classification."""
    n0 = edge_counts["zero"]
    nr = edge_counts["realistic"]
    ns = edge_counts["stress"]
    if n0 == 0:
        return "NO_ZERO_COST_EDGE"
    if nr == n0 and ns == n0:
        return "COST_ROBUST"
    if nr == n0 and ns < n0:
        return "COST_FRAGILE_STRESS"
    return "COST_ELIMINATED"


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

    print("\n=== 3. Leakage review (lookback=5, all cost levels) ===")
    for ticker in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=CANONICAL_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        for level in COST_LEVELS:
            cfg = cost_config(level)
            res = bt.run_bars(list(bars), signals, cfg)
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, cfg.initial_capital)
            total_return = (res.equity_curve[-1] - cfg.initial_capital) / cfg.initial_capital
            gross = sum(abs(f.shares) * f.price for f in res.trades)
            costs = sum(f.commission for f in res.trades) + sum(
                abs(f.shares) * (cfg.slippage_cents / 100.0 + f.price * cfg.slippage_proportional)
                for f in res.trades
            )
            print(f"  {ticker} [{level:9s}]: {bars.n_bars} bars, {len(res.trades)} fills, "
                  f"gross_traded {gross:,.0f}, round-trip cost {costs:,.0f}, "
                  f"total_return {total_return:+.3f} | leakage [PASS]")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost-sensitivity sweep (lookback 3 / 5 / 10 x cost grid) ===")
    print("  Predeclared cost grid: zero / realistic / stress(2x). "
          "Walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 contiguous volatility blocks; "
          "each cost level vs a coin-flip null matched to the same costs.\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    results = {}
    for level in COST_LEVELS:
        cfg = cost_config(level)
        results[level] = {}
        for lb in LOOKBACKS:
            results[level][str(lb)] = {}
            for ticker in ticker_order:
                results[level][str(lb)][ticker] = segment_result(all_tickers[ticker], lb, cfg)
            counts = {k: 0 for k in VERDICT_KEYS}
            for ticker in ticker_order:
                counts[results[level][str(lb)][ticker]["verdict"]] += 1
            n_edge = sum(1 for t in ticker_order if is_positive_edge(results[level][str(lb)][t]))
            print(f"  cost={level:9s} lookback={lb:2d}: verdict_counts={counts} "
                  f"positive_edges={n_edge}/{len(ticker_order)}")

    print("\n  Per-asset profiles (lookback 5 candidate; cost-robustness class):")
    per_asset = {}
    for ticker in ticker_order:
        entry = {"lookbacks": {}}
        for lb in LOOKBACKS:
            lb_entry = {}
            for level in COST_LEVELS:
                lb_entry[level] = results[level][str(lb)][ticker]
            lb_entry["edge_flags"] = {
                level: is_positive_edge(lb_entry[level]) for level in COST_LEVELS}
            lb_entry["median_positive_flags"] = {
                level: all_medians_positive(lb_entry[level]) for level in COST_LEVELS}
            lb_entry["cost_robustness"] = classify_cost(lb_entry["edge_flags"])
            m0 = lb_entry["zero"]["medians"]
            lb_entry["median_deltas"] = {
                level: [round(a - b, 3) for a, b in zip(lb_entry[level]["medians"], m0)]
                for level in COST_LEVELS
            }
            lb_entry["max_abs_median_delta"] = {
                level: max((abs(x) for x in lb_entry["median_deltas"][level]), default=0.0)
                for level in COST_LEVELS
            }
            entry["lookbacks"][str(lb)] = lb_entry
        entry["lookback_edge_counts"] = {
            level: sum(1 for lb in LOOKBACKS if is_positive_edge(results[level][str(lb)][ticker]))
            for level in COST_LEVELS
        }
        entry["lookback_robustness"] = {
            level: ("ROBUST" if entry["lookback_edge_counts"][level] >= 2
                    else ("SENSITIVE" if entry["lookback_edge_counts"][level] == 1 else "NO_EDGE"))
            for level in COST_LEVELS
        }
        per_asset[ticker] = entry
        l5 = entry["lookbacks"][str(CANONICAL_LOOKBACK)]
        med = ", ".join(
            "{}:{:+.3f}".format(level, l5[level]["medians"][0]) for level in COST_LEVELS)
        print("    {:6s} segments={:<8} medians(l5)={:<42s} class={:<18s} "
              "edges={} rob={}".format(
                  ticker, str(l5["zero"]["segments"]), med,
                  l5["cost_robustness"], l5["edge_flags"],
                  entry["lookback_edge_counts"]))
    print("")

    universe = {
        "verdict_counts": {
            level: {str(lb): {
                k: sum(1 for t in ticker_order
                       if results[level][str(lb)][t]["verdict"] == k)
                for k in VERDICT_KEYS}
                for lb in LOOKBACKS}
            for level in COST_LEVELS
        },
        "positive_edge_counts": {
            str(lb): {
                level: sum(1 for t in ticker_order
                           if is_positive_edge(results[level][str(lb)][t]))
                for level in COST_LEVELS}
            for lb in LOOKBACKS
        },
        "positive_edge_counts_l5": {
            level: sum(1 for t in ticker_order
                       if is_positive_edge(results[level][str(CANONICAL_LOOKBACK)][t]))
            for level in COST_LEVELS
        },
        "median_positive_counts": {
            str(lb): {
                level: sum(1 for t in ticker_order
                           if all_medians_positive(results[level][str(lb)][t]))
                for level in COST_LEVELS}
            for lb in LOOKBACKS
        },
        "median_positive_counts_l5": {
            level: sum(1 for t in ticker_order
                       if all_medians_positive(results[level][str(CANONICAL_LOOKBACK)][t]))
            for level in COST_LEVELS
        },
        "lookback_robust_counts": {
            level: sum(1 for t in ticker_order
                       if per_asset[t]["lookback_robustness"][level] == "ROBUST")
            for level in COST_LEVELS
        },
        "ticker_order": ticker_order,
    }
    universe["aggregate_cost_robustness"] = classify_aggregate(
        universe["positive_edge_counts_l5"])
    universe["aggregate_counts"] = dict(universe["positive_edge_counts_l5"])

    print("=== 5. Aggregate cost robustness ===")
    print("  positive edges at lookback 5 (verdict-based): {}".format(
        universe["positive_edge_counts_l5"]))
    print("  strictly-positive segment medians at lookback 5 "
          "(verdict-independent): {}".format(universe["median_positive_counts_l5"]))
    print("  lookback-ROBUST assets (>=2 lookbacks with a positive edge): {}".format(
        universe["lookback_robust_counts"]))
    print("  aggregate cost-robustness class: {}".format(
        universe["aggregate_cost_robustness"]))
    print("")

    # Determinism: rerun the AMZN/JPM subset across all lookbacks and cost levels.
    res2 = {}
    for level in COST_LEVELS:
        cfg = cost_config(level)
        res2[level] = {}
        for lb in LOOKBACKS:
            res2[level][str(lb)] = {
                t: segment_result(all_tickers[t], lb, cfg) for t in ["AMZN", "JPM"]}
    out1 = json.dumps(
        {level: {str(lb): {t: results[level][str(lb)][t] for t in ["AMZN", "JPM"]}
                 for lb in LOOKBACKS} for level in COST_LEVELS},
        sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        canonical_lookback=CANONICAL_LOOKBACK,
        cost_levels=list(COST_LEVELS),
        cost_grid=cost_grid_record(),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        results=results,
        per_asset=per_asset,
        universe=universe,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())