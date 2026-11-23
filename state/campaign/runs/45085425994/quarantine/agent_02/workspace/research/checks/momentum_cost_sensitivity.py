"""Real-data transaction-cost sensitivity of the short-horizon momentum edge.

Objective: the momentum signal class (long the previous N-day return, hold one
day, daily rebalance) is admitted in `state/LEARNING_STATE.md` as *candidate
positive evidence* with REGIME_STABLE positive medians in 7/10 collected assets
at lookback 5 (activation 37318950814) and lookback-robust medians across
3/5/10 (activation 37361000967). That admission was produced with the engine's
default zero-cost `BacktestConfig`, so the decision-relevant open question is
whether the edge survives conservative transaction costs. This check answers
that question on the same 10-asset universe, the same walk-forward windows and
the same past-only volatility blocks as the prior momentum checks.

Predeclared design (fixed a-priori, NOT tuned to any observed result):

* Cost grid (three points; no intermediate values are examined before the run):
    1. `zero`       -- commission 0, slippage 0 (reference: reproduces the
                        zero-cost medians that the existing momentum artifacts
                        report; a mismatch here is a framework defect, not a
                        research result);
    2. `realistic`  -- the repository's established "realistic" cost bundle
                        (examples/ma_crossover.py, examples/ma_crossover_real_data.py,
                        examples/volatility_regime_filter.py):
                        commission_per_trade=2.0, commission_per_share=0.003,
                        slippage_cents=2.0, slippage_proportional=0.0005;
    3. `stress2x`   -- 2x the proportional component of `realistic`
                        (commission_per_trade=4.0, commission_per_share=0.006,
                        slippage_cents=4.0, slippage_proportional=0.0010).
* Lookbacks {3, 5, 10} -- the same set as `momentum_lookback_sweep.py`.
* Walk-forward train=252d / test=84d / warmup=60d / overlap=60d, 4 contiguous
  volatility blocks, min 400 bars per segment -- identical to the prior checks.
* Matched null: a coin-flip sign null run on the same segments *with the same
  cost config*, so the null absorbs the same cost drag. The null is therefore
  expected to be negative whenever costs are material; the diagnostic is the
  candidate-minus-null gap, not the raw candidate level.

A-priori falsification prediction (NOT an observation): the momentum signal
flips between +1 and -1 nearly every bar, so the traded notional per bar is
about 2x initial capital. At `realistic` costs (5 bps proportional + 2 cents
per share per side) a round trip costs roughly 15-25 bps of capital, while the
reported zero-cost medians are 0.05-0.13 log return per *fold* (an 84-bar OOS
segment), i.e. a few bps per bar. If that arithmetic is right, `realistic`
costs should compress the medians toward or below the matched null and the
`realistic`/`stress2x` cells should lose the REGIME_STABLE positive verdicts in
most assets. The opposite outcome (edge survives 2x costs) would be strong
evidence for the candidate; either outcome closes the cell.

SIGNAL-DEFECT NOTICE (discovered during this activation, campaign slot 02):
`research.backtest.momentum_signals` evaluated
``mean(log(closes[i-lookback+1 : i+1]))`` — the mean log *price level* over the
window, not a return. Its sign compares the window's mean price with a fixed
zero level, so on a universe priced far above zero it is almost always +1:
measured one position flip in 4465 bars on 9/10 collected assets, and the
zero-cost medians of `momentum_results.json` are reproduced exactly by an
explicit always-long benchmark on 9/10 assets. The "momentum" cell admitted in
activations 37318950814 / 37361000967 / 37394344201 was therefore an
almost-always-long buy-and-hold bet, not a momentum bet. The function is
corrected in this activation to the past-only return
``closes[i] / closes[i - lookback] - 1``; every figure below is recomputed with
the corrected signal, and the earlier momentum verdicts are retracted in
`state/STATE.md` rather than carried forward.

Cost-robustness classification (predeclared, applied per asset and to the
universe):
* A cost point is EDGE on an asset if the cost-aware verdict is REGIME_STABLE
  and every segment median is positive.
* COST_ROBUST   : `realistic` is EDGE on the asset;
* COST_FRAGILE  : `zero` was EDGE but `realistic` is not EDGE;
* NO_EDGE_ZERO  : `zero` itself is not EDGE (nothing to stress);
* COST_REVERSED : `realistic` medians are uniformly negative (stable losses).

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
REF_LOOKBACK = 5
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

# Predeclared cost grid. Each entry is a full BacktestConfig keyword set so the
# cost model is auditable from the artifact alone.
COST_GRID = {
    "zero": dict(
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    ),
    "realistic": dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    ),
    "stress2x": dict(
        commission_per_trade=4.0,
        commission_per_share=0.006,
        slippage_cents=4.0,
        slippage_proportional=0.0010,
    ),
}
COST_POINTS = ("zero", "realistic", "stress2x")


def cost_config(name, **overrides):
    """BacktestConfig for one predeclared cost point (constant-dollar sizing)."""
    params = dict(COST_GRID[name])
    params.update(overrides)
    return bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, **params)


def segment_result(ticker, bars, lookback, cost_name):
    """Walk-forward regime-stability result for one ticker/lookback/cost point.

    Uses the framework's own `stress_segments`, which forwards the supplied
    `cfg` to both the candidate sweep and the coin-flip null, so the null is
    matched to the same cost model.
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
        cfg=cost_config(cost_name),
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
    """A cost point shows an edge when REGIME_STABLE with uniformly positive medians."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def classify(zero_sr, real_sr):
    """Predeclared cost-robustness classification for one asset/lookback."""
    zero_edge = is_edge(zero_sr)
    real_edge = is_edge(real_sr)
    if not zero_edge:
        return "NO_EDGE_ZERO"
    if real_edge:
        return "COST_ROBUST"
    if all(m < 0 for m in real_sr["medians"]):
        return "COST_REVERSED"
    return "COST_FRAGILE"


def gap(zero_medians, real_medians):
    """Per-segment candidate-minus-null gap for the zero and realistic points."""
    return dict(
        zero=[round(z - n, 3) for z, n in zero_medians],
        realistic=[round(r - n, 3) for r, n in real_medians],
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
        signals = bt.momentum_signals(closes, lookback=REF_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cost_config("realistic"))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], realistic-cost total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "the same signal series is used for every cost point, so cost "
          "differences cannot come from signal differences.")

    print("\n=== 4. Momentum cost-sensitivity sweep (lookbacks 3/5/10 x costs "
          "zero/realistic/stress2x) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, 4 contiguous "
          "volatility blocks, matched coin-flip null under the same costs\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    for name in COST_POINTS:
        c = COST_GRID[name]
        print("  cost point {:9s}: commission_per_trade={:.1f} "
              "commission_per_share={:.3f} slippage_cents={:.1f} "
              "slippage_proportional={:.4f}".format(
                  name, c["commission_per_trade"], c["commission_per_share"],
                  c["slippage_cents"], c["slippage_proportional"]))
    print("")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    results = {lb: {cp: {t: segment_result(t, all_tickers[t], lb, cp)
                         for t in ticker_order}
                    for cp in COST_POINTS}
               for lb in LOOKBACKS}

    per_asset = {}
    class_counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "COST_REVERSED": 0,
                    "NO_EDGE_ZERO": 0}
    verdict_counts = {cp: {lb: {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                                "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
                           for lb in LOOKBACKS}
                      for cp in COST_POINTS}
    for ticker in ticker_order:
        profile = {"cost_cells": {}, "class_by_lookback": {}}
        asset_class = {}
        for lb in LOOKBACKS:
            entry_lb = {}
            for cp in COST_POINTS:
                sr = results[lb][cp][ticker]
                entry_lb[cp] = {
                    "medians": sr["medians"],
                    "null_medians": sr["null_medians"],
                    "candidate_dispersion": sr["candidate_dispersion"],
                    "null_dispersion": sr["null_dispersion"],
                    "n_folds": sr["n_folds"],
                    "segments": sr["segments"],
                    "verdict": sr["verdict"],
                }
                verdict_counts[cp][lb][sr["verdict"]] += 1
            entry_lb["class"] = classify(entry_lb["zero"], entry_lb["realistic"])
            entry_lb["gap"] = gap(
                list(zip(entry_lb["zero"]["medians"], entry_lb["zero"]["null_medians"])),
                list(zip(entry_lb["realistic"]["medians"],
                         entry_lb["realistic"]["null_medians"])),
            )
            asset_class[str(lb)] = entry_lb["class"]
            class_counts[entry_lb["class"]] += 1
            profile["cost_cells"][str(lb)] = entry_lb
        profile["class_by_lookback"] = asset_class
        per_asset[ticker] = profile

    # Reference-lookback summary (lookback 5 = the admitted momentum cell).
    ref_class = {t: per_asset[t]["class_by_lookback"][str(REF_LOOKBACK)]
                 for t in ticker_order}
    ref_counts = {k: sum(1 for v in ref_class.values() if v == k)
                  for k in class_counts}
    zero_ref_edges = sum(
        1 for t in ticker_order
        if per_asset[t]["cost_cells"][str(REF_LOOKBACK)]["zero"]["verdict"] == "REGIME_STABLE"
        and all(m > 0 for m in per_asset[t]["cost_cells"][str(REF_LOOKBACK)]["zero"]["medians"]))
    real_ref_edges = sum(
        1 for t in ticker_order
        if per_asset[t]["cost_cells"][str(REF_LOOKBACK)]["realistic"]["verdict"] == "REGIME_STABLE"
        and all(m > 0 for m in per_asset[t]["cost_cells"][str(REF_LOOKBACK)]["realistic"]["medians"]))

    print("  Reference lookback {} (the momentum cell under test)".format(REF_LOOKBACK))
    print("    zero-cost EDGE assets:       {}/10".format(zero_ref_edges))
    print("    realistic-cost EDGE assets:  {}/10".format(real_ref_edges))
    print("    class counts (all lookbacks): {}".format(class_counts))
    print("    class counts (reference lookback): {}".format(ref_counts))
    print("")
    print("  Per-asset reference-lookback detail:")
    for ticker in ticker_order:
        cell = per_asset[ticker]["cost_cells"][str(REF_LOOKBACK)]
        z = cell["zero"]
        r = cell["realistic"]
        s = cell["stress2x"]
        print("    {:6s} zero={:9s} med={} | real={:9s} med={} | stress2x={:9s} "
              "med={} -> {}".format(
                  ticker, z["verdict"], z["medians"], r["verdict"], r["medians"],
                  s["verdict"], s["medians"],
                  per_asset[ticker]["class_by_lookback"][str(REF_LOOKBACK)]))
    print("")
    print("  Lookback {} medians by cost point (candidate medians; null medians):".format(
        REF_LOOKBACK))
    for ticker in ticker_order:
        cell = per_asset[ticker]["cost_cells"][str(REF_LOOKBACK)]
        z = cell["zero"]
        r = cell["realistic"]
        s = cell["stress2x"]
        print("    {:6s} zero {} / {} | real {} / {} | stress2x {} / {}".format(
            ticker, z["medians"], z["null_medians"],
            r["medians"], r["null_medians"],
            s["medians"], s["null_medians"]))
    print("")

    # Cross-consistency: the zero-cost column must reproduce the existing
    # momentum artifacts at lookback 5 (a framework-alignment gate, not a
    # research result). NOTE: after the 2026-11-22 signal correction this gate
    # is EXPECTED TO FAIL against the pre-correction artifacts, because those
    # artifacts encode the degenerate mean-log-price signal. The comparison is
    # retained deliberately so the discrepancy stays visible and auditable
    # instead of being silently dropped.
    mom_path = ARTIFACT_DIR / "momentum_results.json"
    sweep_path = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"
    cross = {}
    if mom_path.exists() and sweep_path.exists():
        with open(mom_path) as f:
            mom = json.load(f)
        with open(sweep_path) as f:
            swp = json.load(f)
        ok = True
        for ticker in ticker_order:
            pub = mom["universe"]["per_asset"][ticker]
            pub_swp = swp["universe"]["per_asset"][ticker]
            zero = per_asset[ticker]["cost_cells"][str(REF_LOOKBACK)]["zero"]
            same_mom = (zero["medians"] == pub["medians"]
                        and zero["null_medians"] == pub["null_medians"]
                        and zero["verdict"] == pub["verdict"])
            same_swp = (zero["medians"] == pub_swp["lookback_medians"]["5"]
                        and zero["null_medians"] == pub_swp["lookback_null_medians"]["5"])
            cross[ticker] = dict(momentum_results=same_mom,
                                 lookback_sweep=same_swp)
            ok = ok and same_mom and same_swp
        cross["all_match"] = ok
        cross["interpretation"] = (
            "The zero-cost column of this artifact is expected to DISAGREE with "
            "the pre-2026-11-22 momentum artifacts because those artifacts encode "
            "the degenerate mean-log-price signal; the disagreement is the "
            "evidence that the old momentum verdicts described buy-and-hold, not "
            "momentum.")
        print("  cross-consistency vs momentum_results.json + "
              "momentum_lookback_sweep_results.json (zero-cost column): {}".format(
                  "MATCH" if ok else "MISMATCH (expected post signal-correction)"))
        for ticker in ticker_order:
            if not (cross[ticker]["momentum_results"]
                    and cross[ticker]["lookback_sweep"]):
                print("    {}: {}".format(ticker, cross[ticker]))
    else:
        cross = dict(all_match=False, reason="reference artifacts missing")
        print("  cross-consistency gate SKIPPED: reference artifacts missing")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {cp: {t: segment_result(t, tickers[t], lb, cp) for t in target}
                 for cp in COST_POINTS}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {cp: {t: results[lb][cp][t] for t in target}
                            for cp in COST_POINTS}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        reference_lookback=REF_LOOKBACK,
        cost_points={name: COST_GRID[name] for name in COST_POINTS},
        cost_model=dict(
            commission="per trade + per share, charged on |delta shares| at every "
                     "fill (a daily flip from +1 to -1 trades ~2x capital)",
            slippage="fixed cents/share + proportional, added to the close fill "
                     "price (close-only pricing, no open/high/low lookahead)",
            sizing="constant-dollar: target = weight * initial_capital, so a "
                   "round trip is ~2x capital and the cost drag is deterministic",
            null="coin-flip sign null run on the same segments under the same "
                 "cost config, so the matched null absorbs the same cost drag",
        ),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        summary=dict(
            reference_lookback_zero_edge_assets=zero_ref_edges,
            reference_lookback_realistic_edge_assets=real_ref_edges,
            class_counts_all_lookbacks=class_counts,
            class_counts_reference_lookback=ref_counts,
            verdict_counts=verdict_counts,
            ticker_order=ticker_order,
        ),
        cross_consistency=cross,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())