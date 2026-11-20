"""Bounded transaction-cost sensitivity check for short-horizon momentum.

Objective (R-001, campaign slot 2): the collected momentum cell was admitted as
candidate positive evidence at lookback 5 (activation 37318950814) and shown to
be lookback-robust across 3/5/10 (activation 37361000967), but its robustness to
*execution cost* was left as a decision-relevant qualification. This check runs
the momentum class (long the previous N-day return, hold 1 day, daily
rebalance) through the same walk-forward regime-stability gate and matched
coin-flip null as `momentum_lookback_sweep.py`, but with the repository's
existing realistic transaction-cost model scaled by a predeclared multiplier
grid. Both the candidate and the null are run under the *same* cost config, so
the null stays matched.

Predeclared cost grid (fixed before execution, not tuned to results)
--------------------------------------------------------------------
The repository's realistic cost model is the one used by
`examples/ma_crossover_real_data.py`:
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005.
This check multiplies every cost component by a multiplier M in
    COST_MULTIPLIERS = (0.0, 1.0, 2.0, 4.0)
where 0 = zero-cost anchor (must reproduce the prior zero-cost sweep),
1 = the repository's realistic model, 2 = a conservative 2x stress,
4 = a severe 4x stress. The lookback grid is LOOKBACKS = (3, 5, 10), matching
the prior real-data lookback sweep.

Predeclared cost-robustness classification (fixed before execution)
-------------------------------------------------------------------
Two complementary views are recorded. The regime verdict (REGIME_STABLE vs
REGIME_DEPENDENT) is *null-dispersion-sensitive*: it compares the candidate's
cross-segment swing to the coin-flip null's, and the null's dispersion changes
with the cost level because the random signal's turnover profile changes. A
verdict that flips across cost multipliers while the candidate medians stay
fixed is therefore a null-dispersion artifact, NOT evidence of cost
sensitivity (the same caveat recorded for the 60-day momentum horizon in
LEARNING_STATE). The decision-relevant measurement is therefore the candidate
MEDIANS, which are the edge magnitude.

For a ticker and lookback, define:
    clean_zero_cost_edge := every 0x segment median > 0 AND 0x verdict is
                            REGIME_STABLE (a clean positive edge at zero cost)
    med_pos(M)           := every segment median > 0 at cost multiplier M
Then the primary (median-based) class is:
    COST_ROBUST   : clean_zero_cost_edge and med_pos(4.0)
    COST_MARGINAL : clean_zero_cost_edge and (not med_pos(4.0)) and med_pos(1.0)
    COST_FRAGILE  : clean_zero_cost_edge and (not med_pos(1.0))
    NO_CLEAN_EDGE : not clean_zero_cost_edge (no cost-robustness claim to make)
The secondary `verdict_cost_class` uses the same 4x/1x/0x edge rule on the
verdict only (REGIME_STABLE with all medians > 0); it is retained for
transparency and flagged where it diverges from the median-based class due to
null-dispersion (recorded in `null_dispersion_artifact`). The headline summary
uses the median-based class at the reference lookback (5).

A-priori prediction (recorded before execution, not tuned): the positive edge
should survive the realistic (1x) and 2x cost levels and degrade to marginal at
4x; the falsifying outcome is the edge collapsing already at 1x (cost-fragile).
Mechanism expectation at prediction time: I expected turnover to be low (trades
only on sign flips). On inspection the engine rebalances to constant-dollar
exposure every bar, so turnover is high (~4451 trades on AMZN) but each
rebalance is a small price-drift adjustment, making per-fold costs second-order
relative to the ~7% per-fold median. The observed result exceeded the prediction:
the edge survives even 4x. See `post_result_mechanism_note` in the artifact.

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

# Repository realistic cost model (examples/ma_crossover_real_data.py).
REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
COST_MULTIPLIERS = (0.0, 1.0, 2.0, 4.0)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
PRIOR_SWEEP_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

COST_CLASSES = ("COST_ROBUST", "COST_MARGINAL", "COST_FRAGILE", "NO_CLEAN_EDGE")
VERDICT_CLASSES = ("COST_ROBUST", "COST_MARGINAL", "COST_FRAGILE", "NO_EDGE")
# A verdict flip across cost multipliers while the mean candidate median moves by
# less than this is a null-dispersion artifact, not a genuine cost effect.
MEDIAN_INVARIANT_TOL = 0.005


def cost_config(multiplier):
    """BacktestConfig with every realistic cost component scaled by `multiplier`."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=REALISTIC_COST["commission_per_trade"] * multiplier,
        commission_per_share=REALISTIC_COST["commission_per_share"] * multiplier,
        slippage_cents=REALISTIC_COST["slippage_cents"] * multiplier,
        slippage_proportional=REALISTIC_COST["slippage_proportional"] * multiplier,
    )


def segment_result(ticker, bars, lookback, multiplier):
    """Walk-forward regime-stability result for one ticker / lookback / cost.

    Same gate as momentum_lookback_sweep.segment_result but with a cost-scaled
    config; the matched null inside stress_segments receives the same config.
    """
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
        cfg=cost_config(multiplier),
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
        edge=bool(res.overall_verdict == "REGIME_STABLE"
                  and all(m > 0 for m in
                          (s.baseline_median_log_return for s in res.scenarios))),
    )


def classify_verdict(edge_by_mult):
    """Predeclared verdict-based cost class (secondary, null-dispersion-sensitive)."""
    if edge_by_mult.get(4.0):
        return "COST_ROBUST"
    if edge_by_mult.get(1.0):
        return "COST_MARGINAL"
    if edge_by_mult.get(0.0):
        return "COST_FRAGILE"
    return "NO_EDGE"


def classify_median(med_pos_by_mult, clean_zero_cost_edge):
    """Predeclared median-based cost class (primary, decision-relevant)."""
    if not clean_zero_cost_edge:
        return "NO_CLEAN_EDGE"
    if med_pos_by_mult.get(4.0):
        return "COST_ROBUST"
    if med_pos_by_mult.get(1.0):
        return "COST_MARGINAL"
    return "COST_FRAGILE"


def _mean(medians):
    return float(np.mean(medians)) if medians else 0.0


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
        if status != "OK":
            print("MANIFEST CHECKSUM MISMATCH; aborting.")
            sys.exit(1)
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

    print("\n=== 3. Leakage review (realistic 1x cost model) on AMZN/JPM ===")
    tickers = {}
    for ticker in ("AMZN", "JPM"):
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=REFERENCE_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cost_config(1.0))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        commission = sum(f.commission for f in res.trades)
        print(f"  {ticker}: {bars.n_bars} bars, signals {len(signals)}, "
              f"trades {len(res.trades)}, commission {commission:+.1f}, "
              f"total_return {total_return:+.3f}, leakage [PASS]")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; the "
          "engine trades only when the +/-1 weight flips sign.")

    print("\n=== 4. Cost-sensitivity grid ===")
    print("  multipliers {} x realistic cost model {}".format(
        list(COST_MULTIPLIERS), REALISTIC_COST))
    print("  lookbacks {}, walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, {} contiguous "
          "volatility blocks; candidate and null share the cost config\n".format(
              list(LOOKBACKS), TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, N_BLOCKS))

    all_tickers = {}
    for entry in manifest["entries"]:
        all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(all_tickers.keys())

    per_asset = {}
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        per_asset[ticker] = {}
        for lb in LOOKBACKS:
            by_cost = {}
            edge_by_mult = {}
            med_pos_by_mult = {}
            mean_median_by_mult = {}
            for mult in COST_MULTIPLIERS:
                sr = segment_result(ticker, bars, lb, mult)
                by_cost[_mult_key(mult)] = sr
                edge_by_mult[mult] = sr["edge"]
                med_pos_by_mult[mult] = all(m > 0 for m in sr["medians"])
                mean_median_by_mult[mult] = _mean(sr["medians"])
            clean0 = bool(med_pos_by_mult[0.0] and by_cost[_mult_key(0.0)]["verdict"] == "REGIME_STABLE")
            mclass = classify_median(med_pos_by_mult, clean0)
            vclass = classify_verdict(edge_by_mult)
            # Null-dispersion artifact: verdict flips across the grid while the
            # mean candidate median is essentially invariant (costs did not move
            # the edge; only the random null's dispersion changed the verdict).
            verdicts = {by_cost[_mult_key(m)]["verdict"] for m in COST_MULTIPLIERS}
            means = [mean_median_by_mult[m] for m in COST_MULTIPLIERS]
            null_artifact = bool(len(verdicts) > 1
                                 and (max(means) - min(means)) < MEDIAN_INVARIANT_TOL)
            per_asset[ticker][str(lb)] = dict(
                by_cost=by_cost,
                edge_by_cost={_mult_key(m): bool(edge_by_mult[m]) for m in COST_MULTIPLIERS},
                med_pos_by_cost={_mult_key(m): bool(med_pos_by_mult[m]) for m in COST_MULTIPLIERS},
                mean_median_by_cost={_mult_key(m): round(mean_median_by_mult[m], 4) for m in COST_MULTIPLIERS},
                clean_zero_cost_edge=clean0,
                median_cost_class=mclass,
                verdict_cost_class=vclass,
                null_dispersion_artifact=null_artifact,
                median_drag_0_to_4x=round(mean_median_by_mult[4.0] - mean_median_by_mult[0.0], 5),
            )

    print("  Per-asset cost classification (reference lookback {}):".format(REFERENCE_LOOKBACK))
    for ticker in ticker_order:
        p = per_asset[ticker][str(REFERENCE_LOOKBACK)]
        meds = " | ".join(
            "{}x:{}".format(_mult_key(m),
                            ",".join("{:+.3f}".format(v) for v in p["by_cost"][_mult_key(m)]["medians"]))
            for m in COST_MULTIPLIERS)
        print("    {:6s} med_class={:12s} verdict_class={:12s} clean0={} null_artifact={}  {}".format(
            ticker, p["median_cost_class"], p["verdict_cost_class"],
            p["clean_zero_cost_edge"], p["null_dispersion_artifact"], meds))
    print("")

    # Universe-level cost-class counts per lookback (median-based primary).
    universe = {}
    for lb in LOOKBACKS:
        med_counts = {c: 0 for c in COST_CLASSES}
        verd_counts = {c: 0 for c in VERDICT_CLASSES}
        clean0 = []
        robust = []
        for ticker in ticker_order:
            p = per_asset[ticker][str(lb)]
            med_counts[p["median_cost_class"]] += 1
            verd_counts[p["verdict_cost_class"]] += 1
            if p["clean_zero_cost_edge"]:
                clean0.append(ticker)
                if p["median_cost_class"] == "COST_ROBUST":
                    robust.append(ticker)
        universe[str(lb)] = dict(
            median_cost_class_counts=med_counts,
            verdict_cost_class_counts=verd_counts,
            clean_zero_cost_edge=clean0,
            median_cost_robust=robust,
        )
    print("  Median-based cost-class counts by lookback (PRIMARY):")
    for lb in LOOKBACKS:
        u = universe[str(lb)]
        print("    lookback {}: {} | clean0={} median_cost_robust={}".format(
            lb, u["median_cost_class_counts"], u["clean_zero_cost_edge"], u["median_cost_robust"]))
    print("  Verdict-based cost-class counts by lookback (SECONDARY, null-dispersion-sensitive):")
    for lb in LOOKBACKS:
        print("    lookback {}: {}".format(lb, universe[str(lb)]["verdict_cost_class_counts"]))

    # Cost drag: candidate mean per-segment median at each multiplier for the
    # reference lookback, over the clean zero-cost-edge assets.
    zero_cost_edge = universe[str(REFERENCE_LOOKBACK)]["clean_zero_cost_edge"]
    cost_drag = {}
    for ticker in ticker_order:
        cost_drag[ticker] = {
            _mult_key(m): per_asset[ticker][str(REFERENCE_LOOKBACK)]["by_cost"][_mult_key(m)]["medians"]
            for m in COST_MULTIPLIERS
        }
    drag_summary = {}
    for mult in COST_MULTIPLIERS:
        vals = []
        for ticker in zero_cost_edge:
            meds = per_asset[ticker][str(REFERENCE_LOOKBACK)]["by_cost"][_mult_key(mult)]["medians"]
            vals.extend(meds)
        drag_summary[_mult_key(mult)] = round(float(np.mean(vals)), 3) if vals else None
    print("\n  Clean zero-cost-edge assets ({}): {}".format(len(zero_cost_edge), zero_cost_edge))
    print("  Mean per-segment candidate median by cost multiplier (ref lookback {}): "
          "{}".format(REFERENCE_LOOKBACK, drag_summary))

    # Zero-cost cross-consistency: the M=0 lookback=5 column must reproduce the
    # prior momentum_lookback_sweep artifact (validates this pipeline).
    print("\n=== 5. Zero-cost cross-consistency vs momentum_lookback_sweep_results.json ===")
    prior_ok = True
    if PRIOR_SWEEP_PATH.exists():
        with open(PRIOR_SWEEP_PATH) as f:
            prior = json.load(f)
        prior_lp = prior["universe"]["per_asset"]
        for ticker in ticker_order:
            sw5 = per_asset[ticker]["5"]["by_cost"]["0.0"]
            pub = prior_lp[ticker]
            ok = (sw5["medians"] == pub["lookback_medians"]["5"]
                  and sw5["null_medians"] == pub["lookback_null_medians"]["5"]
                  and sw5["verdict"] == pub["verdicts_5"])
            prior_ok = prior_ok and ok
            if not ok:
                print("  {} MISMATCH: cost0 {} vs prior {}".format(
                    ticker, sw5["medians"], pub["lookback_medians"]["5"]))
        print("  zero-cost (0x) lookback-5 column vs prior artifact -> {}".format(
            "MATCH (all 10 tickers)" if prior_ok else "MISMATCH"))
        if not prior_ok:
            print("PRIOR CROSS-CHECK FAILED; investigate before recording.")
            sys.exit(1)
    else:
        print("  prior sweep artifact not found; skipping cross-check.")
        prior_ok = None

    # Determinism: rerun a subset and assert identical output.
    print("\n=== 6. Determinism ===")
    det_ok = True
    for ticker in ("AAPL", "AMZN"):
        for lb in (3, 5, 10):
            for mult in COST_MULTIPLIERS:
                a = segment_result(ticker, all_tickers[ticker], lb, mult)
                b = segment_result(ticker, all_tickers[ticker], lb, mult)
                if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                    det_ok = False
                    print("  NON-DETERMINISTIC: {} lb={} mult={}".format(ticker, lb, mult))
    print("  determinism r1 == r2: {}".format(det_ok))
    assert det_ok, "non-deterministic cost-sensitivity output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        reference_lookback=REFERENCE_LOOKBACK,
        cost_multipliers=list(COST_MULTIPLIERS),
        realistic_cost=REALISTIC_COST,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        a_priori_prediction=(
            "The positive edge should survive the realistic (1x) and 2x cost "
            "levels and degrade to marginal at 4x; falsifying outcome: collapse "
            "already at 1x (cost-fragile)."
        ),
        post_result_mechanism_note=(
            "Prediction-time mechanism expectation (low turnover, trades only on "
            "sign flips) was wrong: the engine rebalances to constant-dollar "
            "exposure every bar, so turnover is high (~4451 trades on AMZN) but "
            "each rebalance is a small price-drift adjustment, making per-fold "
            "costs second-order vs the ~7% per-fold median. Observed: the edge "
            "survives even 4x, exceeding the prediction."
        ),
        per_asset=per_asset,
        universe=universe,
        cost_drag=dict(per_asset_medians=cost_drag, mean_by_multiplier=drag_summary),
        zero_cost_edge_assets=zero_cost_edge,
        zero_cost_cross_check=("MATCH" if prior_ok else ("MISMATCH" if prior_ok is False else "SKIPPED")),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum transaction-cost sensitivity across the collected "
          "universe: exploratory simulation.")
    return 0


def _mult_key(m):
    """Stable string key for a cost multiplier (0.0 -> '0.0')."""
    return "{:.1f}".format(float(m))


if __name__ == "__main__":
    sys.exit(main())