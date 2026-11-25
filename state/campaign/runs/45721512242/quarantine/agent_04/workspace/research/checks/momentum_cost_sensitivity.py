"""Real-data transaction-cost sensitivity of the short-horizon momentum edge.

Objective (focused task R-001): decide whether the short-horizon momentum
candidate admitted in activation 37361000967 (positive REGIME_STABLE edge in
7/10 of the collected universe at lookback 5, robust across lookback 3/5/10 in
magnitude) survives conservative transaction-cost stress. Momentum is a
mechanical, daily-rebalanced 1-day-ahead bet, so its evidence is only
decision-relevant if the positive medians persist once realistic execution
costs are charged.

Design (fixed a-priori, NOT tuned to results):
- Universe: the existing 10 collected tickers (dataset
  yf-ohlcv-universe-2009-to-2026-10-03); no universe expansion.
- Signal: the existing ``momentum_signals`` (long the previous N-day return,
  hold 1 day, daily rebalance); lookbacks {3, 5, 10} (the predeclared set from
  the lookback sweep). No signal-rule tuning.
- Cost grid: the repository's existing realistic cost model
  (``commission_per_trade=2.0``, ``commission_per_share=0.003``,
  ``slippage_cents=2.0``, ``slippage_proportional=0.0005`` — the model used by
  ``examples/ma_crossover_real_data.py`` and ``examples/ma_crossover.py``)
  scaled by a fixed multiplier set {0, 1, 2, 5}. 0x is the zero-cost
  reference (must reproduce the admitted momentum artifact); 1x is the
  realistic model; 2x and 5x are progressively more conservative stress
  levels. Every cost component (per-trade, per-share, fixed and proportional
  slippage) is scaled by the same multiplier, so the grid is one-dimensional
  and monotone in cost.
- Matched null: each cell is judged against the framework's coin-flip sign
  null run on the SAME segments with the SAME cost config (costs are charged
  to the null too), via ``stress_segments`` / ``noise_benchmark``.
- Gate: the existing walk-forward regime-stability gate
  (train=252d/test=84d/warmup=60d/overlap=60d, 4 contiguous volatility blocks,
  min 400 bars/segment) across all 10 tickers, per (lookback, cost) cell.

Cost-robustness classification (predeclared, per asset, reference lookback 5):
A cell shows an EDGE when its verdict is REGIME_STABLE with uniformly positive
segment medians. Let m* be the highest cost multiplier in {0,1,2,5} at which
the asset still shows an EDGE:
- COST_ROBUST     : EDGE at 5x (survives the full stress grid)
- COST_MODERATE   : EDGE at 2x but not 5x
- COST_FRAGILE    : EDGE at 1x (realistic) but not 2x
- ZERO_COST_ONLY  : EDGE at 0x only (does not survive realistic costs)
- NO_EDGE         : no EDGE at any cost level (including zero cost)

The classification is monotone in cost by construction (it keys on the highest
cost level at which an edge remains), so a single noisy non-monotone cell
cannot flip the class. The same rule is applied to every lookback for
transparency, but the reported classification uses lookback 5 (the admitted
candidate).

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
REFERENCE_LOOKBACK = 5

# Repository's existing realistic cost model (examples/ma_crossover*.py).
BASE_REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
# Predeclared, one-dimensional cost grid (multipliers of the realistic model).
COST_MULTIPLIERS = (0.0, 1.0, 2.0, 5.0)

COST_CLASSES = ("COST_ROBUST", "COST_MODERATE", "COST_FRAGILE",
                "ZERO_COST_ONLY", "NO_EDGE")

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
MOM_ARTIFACT = ARTIFACT_DIR / "momentum_results.json"


def cost_config(multiplier: float) -> bt.BacktestConfig:
    """BacktestConfig for a cost multiplier of the realistic model.

    ``multiplier == 0`` yields the zero-cost reference (identical to the
    default ``BacktestConfig()`` used by the admitted momentum run); otherwise
    every cost component is scaled by ``multiplier``.
    """
    m = float(multiplier)
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=0,
        commission_per_trade=BASE_REALISTIC_COST["commission_per_trade"] * m,
        commission_per_share=BASE_REALISTIC_COST["commission_per_share"] * m,
        slippage_cents=BASE_REALISTIC_COST["slippage_cents"] * m,
        slippage_proportional=BASE_REALISTIC_COST["slippage_proportional"] * m,
    )


def cell_is_edge(medians, verdict) -> bool:
    """A cell is an EDGE iff REGIME_STABLE with uniformly positive medians."""
    return verdict == "REGIME_STABLE" and all(m > 0 for m in medians)


def classify_cost_robustness(edge_by_cost):
    """Predeclared cost-robustness class from {multiplier: edge_bool}.

    Keys on the highest cost multiplier at which an EDGE remains, so the class
    is monotone in cost and robust to a single non-monotone noisy cell.
    """
    highest = None
    for m in COST_MULTIPLIERS:
        if edge_by_cost.get(m, False):
            highest = m  # COST_MULTIPLIERS is ascending
    if highest is None:
        return "NO_EDGE"
    if highest >= 5.0:
        return "COST_ROBUST"
    if highest >= 2.0:
        return "COST_MODERATE"
    if highest >= 1.0:
        return "COST_FRAGILE"
    return "ZERO_COST_ONLY"


def run_cell(tickers, lookback, multiplier):
    """Run the regime gate across the universe for one (lookback, cost) cell."""
    summary = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", lookback),),
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cost_config(multiplier),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    out = {}
    for ticker in summary.ticker_order:
        r = summary.assets[ticker]
        meds = [round(s.baseline_median_log_return, 3) for s in r.scenarios]
        nulls = [round(s.noise_median_log_return, 3) for s in r.scenarios]
        out[ticker] = dict(
            segments=[s.name for s in r.scenarios],
            medians=meds,
            null_medians=nulls,
            candidate_dispersion=round(r.candidate_dispersion, 3),
            null_dispersion=round(r.null_dispersion, 3),
            verdict=r.overall_verdict,
            n_folds=r.n_folds,
            is_edge=cell_is_edge(meds, r.overall_verdict),
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

    print("\n=== 3. Leakage review (lookback=5, zero cost) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=REFERENCE_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cost_config(0.0))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    # Full universe.
    all_tickers = {}
    for entry in manifest["entries"]:
        all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(all_tickers.keys())

    print("\n=== 4. Cost-sensitivity grid (lookback x cost multiplier) ===")
    print("  Lookbacks: {} | cost multipliers (x realistic {}): {}".format(
        list(LOOKBACKS), BASE_REALISTIC_COST, list(COST_MULTIPLIERS)))
    print("  Walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, {} contiguous volatility blocks, "
          "matched coin-flip null on the same segments\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, N_BLOCKS))

    grid = {}
    for lb in LOOKBACKS:
        grid[str(lb)] = {}
        for m in COST_MULTIPLIERS:
            key = "{:.1f}".format(m)
            grid[str(lb)][key] = run_cell(all_tickers, lb, m)
            counts = {}
            for t in ticker_order:
                counts[grid[str(lb)][key][t]["verdict"]] = \
                    counts.get(grid[str(lb)][key][t]["verdict"], 0) + 1
            print("  lookback {:>2} cost {:>3}x: {}".format(lb, key, counts))

    # Classification (reference lookback 5) + per-lookback transparency.
    print("\n=== 5. Cost-robustness classification (reference lookback {}) ===".format(
        REFERENCE_LOOKBACK))
    classification = {}
    class_counts = {c: 0 for c in COST_CLASSES}
    for t in ticker_order:
        edge_by_cost = {}
        for m in COST_MULTIPLIERS:
            key = "{:.1f}".format(m)
            edge_by_cost[m] = grid[str(REFERENCE_LOOKBACK)][key][t]["is_edge"]
        cls = classify_cost_robustness(edge_by_cost)
        class_counts[cls] += 1
        # Per-lookback edge profile for transparency (not the reported class).
        per_lb_edge = {}
        for lb in LOOKBACKS:
            per_lb_edge[str(lb)] = {
                "{:.1f}".format(m): grid[str(lb)]["{:.1f}".format(m)][t]["is_edge"]
                for m in COST_MULTIPLIERS
            }
        classification[t] = dict(
            reference_lookback=REFERENCE_LOOKBACK,
            edge_by_cost={"{:.1f}".format(m): edge_by_cost[m] for m in COST_MULTIPLIERS},
            highest_cost_with_edge=max(
                [m for m in COST_MULTIPLIERS if edge_by_cost[m]],
                default=None),
            cost_robustness=cls,
            per_lookback_edge=per_lb_edge,
        )
        print("  {:6s} class={:<14} edge_by_cost={}".format(
            t, cls, classification[t]["edge_by_cost"]))

    print("\n  Cost-robustness counts (lookback {}): {}".format(
        REFERENCE_LOOKBACK, class_counts))

    # Diagnostics (derived from the grid; no new runs). The matched coin-flip
    # null is NOT turnover-matched to momentum (random signs flip far more
    # often than a persistent momentum sign), so charging costs to both
    # degrades the null faster than the candidate. These diagnostics expose
    # that confound so the null-relative classification above is not read as a
    # clean cost-robustness test: (a) how much the candidate medians actually
    # move from 0x to 5x, and (b) how unstable the null medians are.
    print("\n=== 6. Diagnostics: candidate cost-sensitivity vs null instability ===")
    diagnostics = {}
    for t in ticker_order:
        c0 = grid[str(REFERENCE_LOOKBACK)]["0.0"][t]["medians"]
        c5 = grid[str(REFERENCE_LOOKBACK)]["5.0"][t]["medians"]
        cand_change = max(abs(a - b) for a, b in zip(c5, c0)) if c0 else 0.0
        nulls_all = []
        for m in COST_MULTIPLIERS:
            key = "{:.1f}".format(m)
            nulls_all.extend(grid[str(REFERENCE_LOOKBACK)][key][t]["null_medians"])
        diagnostics[t] = dict(
            candidate_medians_0x=c0,
            candidate_medians_5x=c5,
            max_abs_candidate_change_0_to_5x=round(cand_change, 3),
            null_min=round(min(nulls_all), 3),
            null_max=round(max(nulls_all), 3),
            null_range=round(max(nulls_all) - min(nulls_all), 3),
        )
        print("  {:6s} cand_change(0->5x)={:.3f}  null_range={:.3f} "
              "[null_min={:+.3f}, null_max={:+.3f}]".format(
                  t, cand_change, diagnostics[t]["null_range"],
                  diagnostics[t]["null_min"], diagnostics[t]["null_max"]))
    n_cost_insensitive = sum(
        1 for t in ticker_order
        if diagnostics[t]["max_abs_candidate_change_0_to_5x"] <= 0.01)
    print("  assets whose candidate medians move <= 0.01 from 0x to 5x: "
          "{}/{}".format(n_cost_insensitive, len(ticker_order)))
    print("  interpretation: candidate medians are near cost-invariant for "
          "strong-edge assets; the null is NOT turnover-matched and becomes "
          "unstable under cost, so the null-relative verdict/classification is "
          "confounded and must be read alongside these diagnostics.")

    # Determinism: rerun the target subset across the full grid and compare.
    res2 = {}
    for lb in LOOKBACKS:
        res2[str(lb)] = {}
        for m in COST_MULTIPLIERS:
            key = "{:.1f}".format(m)
            res2[str(lb)][key] = run_cell(tickers, lb, m)
    out1 = json.dumps({lb: {k: grid[lb][k][t] for t in target}
                       for lb in grid for k in grid[lb]}, sort_keys=True)
    out2 = json.dumps({lb: {k: res2[lb][k][t] for t in target}
                       for lb in res2 for k in res2[lb]}, sort_keys=True)
    det = out1 == out2
    print("\ndeterminism: r1 == r2: {}".format(det))
    assert det, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        reference_lookback=REFERENCE_LOOKBACK,
        cost_multipliers=list(COST_MULTIPLIERS),
        base_realistic_cost=BASE_REALISTIC_COST,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        ticker_order=ticker_order,
        grid=grid,
        classification=classification,
        diagnostics=diagnostics,
        universe_summary=dict(
            cost_robustness_counts=class_counts,
            reference_lookback=REFERENCE_LOOKBACK,
            n_candidate_cost_insensitive=n_cost_insensitive,
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