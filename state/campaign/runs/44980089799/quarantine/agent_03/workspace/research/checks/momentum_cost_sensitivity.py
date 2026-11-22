"""Real-data transaction-cost sensitivity of the short-horizon momentum edge.

Objective (campaign contract R-001): test whether the short-horizon momentum
edge (admitted as candidate positive evidence in activations 37318950814 /
37361000967) survives conservative transaction-cost stress on the collected
real-data universe, for lookback 3 / 5 / 10, under the repository's existing
realistic cost model and its matched coin-flip null.

This check is bounded and predeclared (no tuning after seeing results):

- Universe: the existing 10-ticker collected universe (no expansion).
- Lookbacks: {3, 5, 10} (the same bounded set as the lookback sweep).
- Cost grid, defined relative to the repository's existing realistic cost model
  used in ``examples/ma_crossover_real_data.py`` (commission_per_trade=2.0,
  commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005):

    zero    : no commission / slippage (reproduces the zero-cost baseline)
    base    : the repository's existing realistic cost model
    stress2x: 2x the base fixed + proportional costs (conservative stress)

- Walk-forward: train=252 / test=84 / warmup=60 / overlap=60; 4 contiguous
  volatility blocks; min 400 bars/segment; each lookback compared against a
  coin-flip null on the same segments (the framework's own matched null,
  ``noise_benchmark``, via ``stress_segments``).
- No leverage: ``target_exposure`` stays 1.0 (gross exposure = 1.0x cash); the
  signal rule is not tuned after seeing cost results.

Cost-robustness classification is based on the CANDIDATE's own per-segment
median log returns as costs rise (the direct measure of whether the edge
survives), because the framework's coin-flip null is turnover-asymmetric: its
discrete {-1,0,+1} jumps trade far larger notional than the momentum signal's
small constant-dollar rebalance trades, so the null collapses under costs and
the candidate-minus-null gap widens as an artifact. That confound is recorded
explicitly rather than read as robustness.

The ``zero`` column must reproduce ``momentum_lookback_sweep_results.json``
exactly; that cross-consistency is verified independently in
``verify_momentum_cost_sensitivity.py``.

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
# Tolerance (log-return units) below which the candidate's per-segment medians
# are treated as unchanged by costs.
COST_FLAT_TOL = 0.005

# Predeclared cost grid. ``base`` is the repository's existing realistic cost
# model (examples/ma_crossover_real_data.py); ``stress2x`` doubles the fixed and
# proportional costs as a conservative stress level.
COST_LEVELS = {
    "zero": dict(
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    ),
    "base": dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    ),
    "stress2x": dict(
        commission_per_trade=4.0,
        commission_per_share=0.006,
        slippage_cents=4.0,
        slippage_proportional=0.001,
    ),
}
COST_ORDER = ("zero", "base", "stress2x")

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cost_cfg(level):
    return bt.BacktestConfig(initial_capital=1e6, **COST_LEVELS[level])


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, lookback, and cost
    level, computed through the existing ``stress_segments`` machinery."""
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


def is_edge(sr):
    """A positive REGIME_STABLE edge: uniformly positive segment medians."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def all_positive(sr):
    return all(m > 0 for m in sr["medians"])


def classify_candidate(edge_zero, base_positive, stress_positive, max_change):
    """Predeclared cost-robustness classification from the candidate's own
    per-segment medians as costs rise.

    NO_EDGE_AT_ZERO         : no positive REGIME_STABLE edge even at zero cost.
    COST_ELIMINATED         : a zero-cost edge asset has a non-positive segment
                              median at realistic (base) cost.
    COST_STRESS_ELIMINATED  : edges survive base cost but not 2x stress.
    COST_ROBUST             : edges survive base and 2x stress and every
                              per-segment median moves by <= COST_FLAT_TOL.
    COST_DEGRADED           : edges survive but at least one median moves by
                              more than COST_FLAT_TOL.
    """
    if not edge_zero:
        return "NO_EDGE_AT_ZERO"
    if base_positive is False:
        return "COST_ELIMINATED"
    if stress_positive is False:
        return "COST_STRESS_ELIMINATED"
    if max_change <= COST_FLAT_TOL:
        return "COST_ROBUST"
    return "COST_DEGRADED"


def turnover_profile(bars, lookback):
    """Full-sample trade/commission profile for the candidate vs a matched
    coin-flip null, at the base realistic cost.

    Costs do not change the number of trades (orders follow signal changes), so
    this is computed once per (ticker, lookback). It documents the turnover
    asymmetry between the momentum candidate and the random null: the null's
    discrete {-1,0,+1} jumps trade far larger notional than the momentum
    signal's small constant-dollar rebalance trades.
    """
    closes = bars.closes_array()
    n = len(closes)
    cfg = cost_cfg("base")
    cand = bt.run_bars(list(bars), bt.momentum_signals(closes, lookback=lookback), cfg)
    null = bt.run_bars(list(bars), bt.random_signals(n, seed=SEED), cfg)
    return dict(
        candidate_trades=len(cand.trades),
        null_trades=len(null.trades),
        candidate_commission=round(sum(f.commission for f in cand.trades), 1),
        null_commission=round(sum(f.commission for f in null.trades), 1),
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
              f"leakage [PASS], zero-cost total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum cost-sensitivity sweep (lookback 3 / 5 / 10 x cost grid) ===")
    print("  cost levels: " + ", ".join(COST_ORDER))
    for name in COST_ORDER:
        print(f"    {name}: {COST_LEVELS[name]}")
    print("  Params: walk-forward train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 contiguous volatility blocks, "
          "each lookback vs a coin-flip null on the same segments\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # per_cell[lookback][cost][ticker] = segment_result
    per_cell = {}
    for lb in LOOKBACKS:
        per_cell[lb] = {}
        for level in COST_ORDER:
            cfg = cost_cfg(level)
            per_cell[lb][level] = {
                t: segment_result(t, all_tickers[t], lb, cfg) for t in ticker_order
            }

    # Turnover diagnostic (cost-independent count; base-cost commission).
    turnover = {t: {str(lb): turnover_profile(all_tickers[t], lb) for lb in LOOKBACKS}
                for t in ticker_order}

    # Classify cost robustness per lookback from the candidate's own medians.
    cost_robustness = {}
    for lb in LOOKBACKS:
        zero = per_cell[lb]["zero"]
        base = per_cell[lb]["base"]
        stress = per_cell[lb]["stress2x"]

        edge_zero = [t for t in ticker_order if is_edge(zero[t])]
        base_positive = (all(all_positive(base[t]) for t in edge_zero)
                         if edge_zero else None)
        stress_positive = (all(all_positive(stress[t]) for t in edge_zero)
                           if edge_zero else None)

        def max_abs_median_change(cost):
            m = 0.0
            for t in edge_zero:
                for a, b in zip(zero[t]["medians"], per_cell[lb][cost][t]["medians"]):
                    m = max(m, abs(a - b))
            return round(m, 4)

        max_change_base = max_abs_median_change("base")
        max_change_stress = max_abs_median_change("stress2x")
        cls = classify_candidate(edge_zero, base_positive, stress_positive,
                                 max_change_stress)

        # Framework (candidate-vs-null) edge counts, reported for transparency
        # but flagged as confounded by null turnover asymmetry.
        edge_counts = {
            level: sum(1 for t in ticker_order if is_edge(per_cell[lb][level][t]))
            for level in COST_ORDER
        }
        gaps = {}
        for level in COST_ORDER:
            per_ticker_gap = []
            for t in edge_zero or ticker_order:
                sr = per_cell[lb][level][t]
                per_ticker_gap.append(
                    float(np.mean(sr["medians"])) - float(np.mean(sr["null_medians"])))
            gaps[level] = round(float(np.median(per_ticker_gap)), 4)

        # Turnover asymmetry across the zero-cost edge assets.
        cand_comm = [turnover[t][str(lb)]["candidate_commission"] for t in edge_zero]
        null_comm = [turnover[t][str(lb)]["null_commission"] for t in edge_zero]
        cand_comm_med = float(np.median(cand_comm)) if cand_comm else None
        null_comm_med = float(np.median(null_comm)) if null_comm else None
        ratio = (round(null_comm_med / cand_comm_med, 1)
                 if cand_comm_med else None)

        cost_robustness[str(lb)] = dict(
            classification=cls,
            edge_assets_zero=edge_zero,
            edge_counts_framework=edge_counts,
            candidate_max_abs_median_change={"base": max_change_base,
                                             "stress2x": max_change_stress},
            candidate_all_positive={"base": base_positive,
                                    "stress2x": stress_positive},
            candidate_minus_null_gap=gaps,
            matched_null_confound=dict(
                candidate_commission_median=cand_comm_med,
                null_commission_median=null_comm_med,
                null_over_candidate_commission_ratio=ratio,
                note=("coin-flip null turnover >> candidate turnover; null "
                      "collapses under costs so the candidate-minus-null gap "
                      "widening is an artifact, not evidence of robustness"),
            ),
        )

    print("  Cost-robustness classification by lookback (candidate absolute medians):")
    for lb in LOOKBACKS:
        cr = cost_robustness[str(lb)]
        print("    lookback {}: edge_assets(zero)={} classification={}".format(
            lb, cr["edge_assets_zero"], cr["classification"]))
        print("      candidate max |median change| base/stress2x = {} / {}".format(
            cr["candidate_max_abs_median_change"]["base"],
            cr["candidate_max_abs_median_change"]["stress2x"]))
        print("      candidate all-positive base/stress2x = {} / {}".format(
            cr["candidate_all_positive"]["base"],
            cr["candidate_all_positive"]["stress2x"]))
        print("      framework edge_counts zero/base/stress2x = {}/{}/{} (CONFOUNDED)".format(
            cr["edge_counts_framework"]["zero"],
            cr["edge_counts_framework"]["base"],
            cr["edge_counts_framework"]["stress2x"]))
        print("      null/candidate commission ratio = {} (turnover asymmetry)".format(
            cr["matched_null_confound"]["null_over_candidate_commission_ratio"]))

    print("\n  Per-asset candidate medians by cost level (lookback 5):")
    for t in ticker_order:
        cells = []
        for level in COST_ORDER:
            sr = per_cell[5][level][t]
            cells.append("{}={}".format(level, sr["medians"]))
        print("    {:6s} {}".format(t, " | ".join(cells)))

    # Determinism: rerun the AMZN/JPM subset for lookback=5, base, and assert
    # identical output.
    res2 = {t: segment_result(t, tickers[t], 5, cost_cfg("base")) for t in target}
    out1 = json.dumps({t: per_cell[5]["base"][t] for t in target}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("\ndeterminism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_order=list(COST_ORDER),
        cost_levels={name: dict(COST_LEVELS[name]) for name in COST_ORDER},
        cost_flat_tol=COST_FLAT_TOL,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_robustness=cost_robustness,
        turnover=turnover,
        per_cell={str(lb): per_cell[lb] for lb in LOOKBACKS},
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum transaction-cost sensitivity across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())