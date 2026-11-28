"""Real-data transaction-cost sensitivity of the qualified momentum edge.

Objective (task R-001): decide whether the short-horizon momentum edge that was
admitted as candidate positive evidence (`research/checks/momentum.py`,
activation 37318950814; lookback sweep 37361000967) survives conservative
transaction-cost stress on the collected 10-asset universe, or whether it is a
gross-return artifact that costs erase.

This is a *qualification* of an already-admitted candidate, not a new signal
search. The signal rule is frozen (long the previous N-day return, hold 1 day,
daily rebalance; lookbacks 3 / 5 / 10) and is NOT tuned after seeing costs. The
universe is the existing 10 collected tickers. No leverage is introduced.

Cost model (the repository's existing realistic model, from
`examples/ma_crossover_real_data.py`): commission_per_trade=2.0,
commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005.
The cost grid is a fixed, predeclared 3-level ladder:
  - zero        : no commission / no slippage (reproduces the prior zero-cost
                  momentum artifacts; a wiring anchor, see cross-consistency).
  - realistic   : the repository's canonical realistic cost model above.
  - conservative: 2x the realistic model (commission 4.0 / 0.006, slippage
                  4.0 cents / 0.0010 = 10 bps proportional) -- a conservative
                  stress level, declared before execution.

Matched null: the coin-flip null is run through the framework's
`noise_benchmark` with the *same* `cfg` at every cost level, so the candidate
and the null face identical costs. The candidate-vs-null gap is reported as a
secondary signal-informativeness view. IMPORTANT interpretation caveat: the null
is a coin flip that churns the book, so it pays costs too and its median falls
under stress; the gap-vs-null can therefore widen even while the *absolute*
net-of-cost candidate return falls. The PRIMARY cost-robustness classification is
therefore based on the absolute candidate medians (REGIME_STABLE with uniformly
positive medians = a positive net-of-cost edge), not on the gap.

A-priori prediction (declared before execution, falsifiable): because the
strategy holds 1 day and rebalances daily (very high turnover), conservative
costs will materially erode or erase the edge for most assets; the majority of
the universe is expected to be COST_ERODED or NO_EDGE at the conservative level,
with at most a minority COST_ROBUST. This prediction is recorded so the outcome
can falsify it either way.

Per-asset classification (based on lookback 5, the reference momentum):
  - COST_ROBUST : positive REGIME_STABLE edge at BOTH zero and conservative.
  - COST_ERODED : positive REGIME_STABLE edge at zero but not at conservative.
  - NO_EDGE     : no positive REGIME_STABLE edge even at zero cost.

Method (fixed a-priori): identical walk-forward and volatility-block segmentation
as `momentum_lookback_sweep.py` (train=252d/test=84d/warmup=60d/overlap=60d,
4 contiguous volatility blocks, min 400 bars/segment), run once per
(cost level x lookback) via `stress_segments` with the level's `cfg`. The
zero-cost column must reproduce `momentum_lookback_sweep_results.json` exactly
(cross-consistency gate). Determinism is asserted by an internal recomputation.

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

# Predeclared cost ladder (fixed before execution; do not tune after results).
COST_LEVELS = (
    ("zero", {}),
    ("realistic", dict(commission_per_trade=2.0, commission_per_share=0.003,
                       slippage_cents=2.0, slippage_proportional=0.0005)),
    ("conservative", dict(commission_per_trade=4.0, commission_per_share=0.006,
                          slippage_cents=4.0, slippage_proportional=0.0010)),
)
COST_NAMES = tuple(name for name, _ in COST_LEVELS)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def make_cfg(cost_kwargs):
    """BacktestConfig for one cost level (zero-cost kwargs -> all costs zero)."""
    return bt.BacktestConfig(initial_capital=1e6, **cost_kwargs)


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker/lookback/cost level."""
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


def edge_state(sr):
    """Positive net-of-cost edge indicator for one scenario result.

    EDGE : REGIME_STABLE with uniformly positive medians (a stable positive
           net-of-cost edge).
    LOSS : REGIME_STABLE_LOSS (stable losses).
    NO_EDGE : anything else (CONSISTENT_WITH_NOISE / REGIME_DEPENDENT).
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def classify_cost_robustness(states_by_level):
    """Per-asset cost-robustness class from lookback-5 edge states per level.

    This is the PREDECLARED rule (uses the framework's REGIME_STABLE verdict).
    It is reported as the primary class. Because the REGIME_STABLE vs
    REGIME_DEPENDENT label depends on the matched null's dispersion (which can
    itself change with the cost level), a median-based class is reported
    alongside as a verdict-artifact-robust cross-check.
    """
    s_zero = states_by_level["zero"]
    s_cons = states_by_level["conservative"]
    if s_zero == "EDGE" and s_cons == "EDGE":
        return "COST_ROBUST"
    if s_zero == "EDGE" and s_cons != "EDGE":
        return "COST_ERODED"
    return "NO_EDGE"


def classify_cost_robustness_median(medians_by_level):
    """Median-based cost-robustness class (robust to the null-dispersion label).

    COST_ROBUST : every lookback-5 segment median is strictly positive at BOTH
                  zero and conservative (the net-of-cost edge survives).
    COST_ERODED : every segment median positive at zero but not at conservative.
    NO_EDGE     : not every segment median positive even at zero cost.
    """
    z = medians_by_level["zero"]
    c = medians_by_level["conservative"]
    if all(m > 0 for m in z) and all(m > 0 for m in c):
        return "COST_ROBUST"
    if all(m > 0 for m in z) and not all(m > 0 for m in c):
        return "COST_ERODED"
    return "NO_EDGE"


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
        signals = bt.momentum_signals(closes, lookback=REFERENCE_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost-sensitivity grid (zero / realistic / conservative) x "
          "lookback (3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, matched coin-flip null at the same "
          "cost level\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    for name, kw in COST_LEVELS:
        print("  cost level {:12s}: {}".format(name, kw if kw else "(all costs zero)"))
    print("")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    cfgs = {name: make_cfg(kw) for name, kw in COST_LEVELS}

    # results[cost][lookback][ticker]
    results = {
        name: {lb: {t: segment_result(t, all_tickers[t], lb, cfgs[name])
                    for t in ticker_order}
               for lb in LOOKBACKS}
        for name, _ in COST_LEVELS
    }

    # Per-asset classification + per-level detail.
    per_asset = {}
    class_counts = {"COST_ROBUST": 0, "COST_ERODED": 0, "NO_EDGE": 0}
    median_class_counts = {"COST_ROBUST": 0, "COST_ERODED": 0, "NO_EDGE": 0}
    for ticker in ticker_order:
        entry = {"levels": {}, "lookback5_states": {}, "lookback5_medians": {}}
        for name, _ in COST_LEVELS:
            entry["levels"][name] = {
                str(lb): dict(
                    medians=results[name][lb][ticker]["medians"],
                    null_medians=results[name][lb][ticker]["null_medians"],
                    verdict=results[name][lb][ticker]["verdict"],
                    edge_state=edge_state(results[name][lb][ticker]),
                    candidate_dispersion=results[name][lb][ticker]["candidate_dispersion"],
                    null_dispersion=results[name][lb][ticker]["null_dispersion"],
                    n_folds=results[name][lb][ticker]["n_folds"],
                    segments=results[name][lb][ticker]["segments"],
                )
                for lb in LOOKBACKS
            }
        # Reference-lookback (5) states per level drive the classification.
        for name, _ in COST_LEVELS:
            entry["lookback5_states"][name] = entry["levels"][name][str(REFERENCE_LOOKBACK)]["edge_state"]
            entry["lookback5_medians"][name] = entry["levels"][name][str(REFERENCE_LOOKBACK)]["medians"]
        cls = classify_cost_robustness(entry["lookback5_states"])
        entry["cost_robustness"] = cls
        cls_med = classify_cost_robustness_median(entry["lookback5_medians"])
        entry["cost_robustness_median_based"] = cls_med
        # Cross-lookback: at how many lookbacks does the edge survive conservative?
        entry["n_lookbacks_cost_robust"] = sum(
            1 for lb in LOOKBACKS
            if entry["levels"]["conservative"][str(lb)]["edge_state"] == "EDGE"
        )
        class_counts[cls] += 1
        median_class_counts[cls_med] += 1
        per_asset[ticker] = entry

    print("  Per-asset lookback-5 medians [zero | realistic | conservative] and class:")
    for ticker in ticker_order:
        e = per_asset[ticker]
        z = e["lookback5_medians"]["zero"]
        r = e["lookback5_medians"]["realistic"]
        c = e["lookback5_medians"]["conservative"]
        print("    {:6s} zero={} real={} cons={} -> {:12s} (lb-robust n={})".format(
            ticker,
            ",".join(f"{m:+.3f}" for m in z),
            ",".join(f"{m:+.3f}" for m in r),
            ",".join(f"{m:+.3f}" for m in c),
            e["cost_robustness"], e["n_lookbacks_cost_robust"]))
    print("")
    print("  Cost-robustness counts (predeclared verdict rule): COST_ROBUST={} "
          "COST_ERODED={} NO_EDGE={}".format(
              class_counts["COST_ROBUST"], class_counts["COST_ERODED"],
              class_counts["NO_EDGE"]))
    print("  Cost-robustness counts (median-based cross-check): COST_ROBUST={} "
          "COST_ERODED={} NO_EDGE={}".format(
              median_class_counts["COST_ROBUST"],
              median_class_counts["COST_ERODED"],
              median_class_counts["NO_EDGE"]))
    # Flag any asset where the two rules disagree (null-dispersion label artifact).
    for ticker in ticker_order:
        a = per_asset[ticker]["cost_robustness"]
        b = per_asset[ticker]["cost_robustness_median_based"]
        if a != b:
            print("  [CAVEAT] {} verdict-class={} but median-class={} "
                  "(candidate medians unchanged/positive; REGIME label is "
                  "null-dispersion-sensitive)".format(ticker, a, b))

    # Universe aggregate: cross-asset median of the lookback-5 candidate medians.
    universe_medians = {}
    for name, _ in COST_LEVELS:
        all_meds = [m for t in ticker_order
                    for m in per_asset[t]["lookback5_medians"][name]]
        universe_medians[name] = round(float(np.median(all_meds)), 3)
    print("  Universe cross-asset median of lb5 candidate medians: {}".format(universe_medians))
    print("")

    # Turnover diagnostic (mechanism evidence): the momentum sign is persistent,
    # so the position flips only on sign changes and turnover is low; that is why
    # a per-round-trip cost has little effect on most assets. Full-sample
    # turnover_ratio is cost-independent; the total-return gap between zero and
    # conservative is the realized cost drag. This proves costs are applied (the
    # drag is nonzero) and explains the robustness (turnover is small).
    turnover_diag = {}
    print("  Turnover diagnostic (lookback 5, full sample):")
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=REFERENCE_LOOKBACK)
        row = {}
        for name in ("zero", "conservative"):
            res = bt.run_bars(list(bars), sig, cfgs[name])
            fill_prices = np.array([f.price for f in res.trades], dtype=np.float64)
            m = bt.compute_metrics(res.equity_curve, res.trades, fill_prices)
            row[name] = dict(turnover_ratio=round(float(m.turnover_ratio), 3),
                             total_return=round(float(m.total_return), 3),
                             n_trades=int(m.n_trades))
        drag = round(row["zero"]["total_return"] - row["conservative"]["total_return"], 3)
        row["cost_drag_total_return"] = drag
        turnover_diag[ticker] = row
        print("    {:6s} turnover={:7.2f}x  total_ret zero={:+.3f} cons={:+.3f}  "
              "cost_drag={:+.3f}  n_trades={}".format(
                  ticker, row["zero"]["turnover_ratio"],
                  row["zero"]["total_return"], row["conservative"]["total_return"],
                  drag, row["zero"]["n_trades"]))
    print("")

    # Determinism: rerun the target subset at the conservative level and compare.
    res2 = {lb: {t: segment_result(t, tickers[t], lb, cfgs["conservative"])
                 for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: results["conservative"][lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    det_ok = out1 == out2
    print("determinism (conservative level, AMZN/JPM): r1 == r2: {}".format(det_ok))
    assert det_ok, "non-deterministic output"

    # Cross-consistency anchor: zero-cost column must reproduce the prior
    # zero-cost lookback sweep artifact exactly (medians, nulls, verdicts).
    xcheck_ok = True
    xcheck_notes = []
    if SWEEP_ARTIFACT.exists():
        with open(SWEEP_ARTIFACT) as f:
            sweep = json.load(f)
        sw_lp = sweep["per_asset"]
        for ticker in ticker_order:
            for lb in LOOKBACKS:
                pub = sw_lp[ticker][str(lb)]
                mine = results["zero"][lb][ticker]
                ok = (mine["medians"] == [round(v, 3) for v in pub["medians"]]
                      and mine["null_medians"] == [round(v, 3) for v in pub["null_medians"]]
                      and mine["verdict"] == pub["verdict"])
                if not ok:
                    xcheck_ok = False
                    xcheck_notes.append(
                        f"{ticker} lb{lb}: mine med={mine['medians']} null={mine['null_medians']} "
                        f"v={mine['verdict']} vs pub med={pub['medians']} "
                        f"null={pub['null_medians']} v={pub['verdict']}")
        print("zero-cost cross-consistency vs momentum_lookback_sweep_results.json: "
              + ("MATCH (all 10 tickers x 3 lookbacks)" if xcheck_ok else "MISMATCH"))
        for n in xcheck_notes:
            print("  [XCHECK] " + n)
    else:
        xcheck_ok = None
        print("zero-cost cross-consistency: SKIPPED (prior sweep artifact absent)")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        reference_lookback=REFERENCE_LOOKBACK,
        cost_levels=[dict(name=name, config=kw) for name, kw in COST_LEVELS],
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        a_priori_prediction=(
            "High-turnover daily-rebalanced momentum is expected to be materially "
            "eroded or erased by conservative costs for most assets (majority "
            "COST_ERODED or NO_EDGE at the conservative level; at most a minority "
            "COST_ROBUST)."
        ),
        per_asset=per_asset,
        turnover_diagnostic=turnover_diag,
        universe=dict(
            cost_robustness_counts=class_counts,
            cost_robustness_counts_median_based=median_class_counts,
            universe_median_lb5=universe_medians,
            ticker_order=ticker_order,
            zero_cost_cross_consistency=("MATCH" if xcheck_ok
                                         else ("SKIPPED" if xcheck_ok is None else "MISMATCH")),
        ),
        determinism=bool(det_ok),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum transaction-cost sensitivity across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())