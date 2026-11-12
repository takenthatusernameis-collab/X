"""Bounded cost-sensitivity check for short-horizon momentum (lookback 3/5/10).

Primary question
----------------
Does the short-horizon momentum edge survive conservative transaction-cost
stress on the collected real-data universe?

Two signal variants are evaluated:

- ``existing``: ``research.backtest.momentum_signals`` -- the framework
  implementation that produced the admitted momentum evidence
  (``state/check_artifacts/momentum_results.json`` and the lookback sweep).
  DIAGNOSTIC FINDING (recorded in the artifact and verified independently):
  this function computes ``np.mean(np.log(closes[i-lookback+1:i+1]))`` -- the
  mean log PRICE LEVEL, not the mean log RETURN. Because every adjusted close
  in the collected universe is above 1.0, that quantity is positive for every
  lookback, so the emitted weight is constant +1 (always long) on 9/10 tickers
  (NVDA is mostly long). The admitted "momentum" evidence therefore measures an
  always-long large-cap drift/beta position, not short-horizon momentum.

- ``corrected``: a local, past-only momentum signal fixed a-priori to the
  documented intent (long when ``closes[i] > closes[i-lookback]``, i.e. the
  sign of the previous ``lookback``-bar price change). This is the genuine
  short-horizon momentum rule the primary question is about. It is defined
  here, in the check, and does NOT mutate the framework; correcting the
  framework function and re-running the admitted cells is recorded as a
  candidate task, not performed in this activation.

Cost grid (bounded, fixed before any cost result is seen), using the
repository's canonical realistic model (examples/ma_crossover_real_data.py and
examples/ma_crossover.py):

- ``zero``        : no costs (reproduces the prior zero-cost baseline)
- ``realistic``   : commission_per_trade=2.0, commission_per_share=0.003,
                    slippage_cents=2.0, slippage_proportional=0.0005
- ``conservative``: 2x realistic (4.0 / 0.006 / 4.0 / 0.001)

For every (signal, lookback, cost, ticker) the rule is run through the same
walk-forward regime-stability gate as the prior momentum checks
(train=252/test=84/warmup=60/overlap=60, 4 contiguous volatility blocks, min
400 bars/segment) with a MATCHED coin-flip null run under the SAME cost config
on the same segments. Determinism is asserted on a target subset, and the
``existing``/zero-cost/lookback=5 column is cross-checked against the prior
``momentum_lookback_sweep_results.json``.

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

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
PRIOR_SWEEP_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

# Predeclared bounded cost grid (repository's canonical realistic model).
COST_LEVELS = {
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.001),
}
COST_ORDER = ("zero", "realistic", "conservative")


def cost_config(level: str) -> bt.BacktestConfig:
    c = COST_LEVELS[level]
    return bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARM,
        commission_per_trade=c["commission_per_trade"],
        commission_per_share=c["commission_per_share"],
        slippage_cents=c["slippage_cents"],
        slippage_proportional=c["slippage_proportional"],
    )


def corrected_momentum_signals(closes, lookback: int = 5):
    """Past-only short-horizon momentum: long when the previous ``lookback``-bar
    price change is positive, short when negative, neutral before the window.

    This matches the documented intent of ``momentum_signals`` (long the previous
    N-day return) without its mean-log-price defect.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        up = float(closes[i]) > float(closes[i - lookback])
        out[i] = bt.Signal(date=i + 1, weight=1.0 if up else -1.0)
    return out


SIGNAL_VARIANTS = {
    "existing": bt.momentum_signals,
    "corrected": corrected_momentum_signals,
}


def weight_counts(signal, start):
    vals = [s.weight for s in signal[start:]]
    return dict(
        n=len(vals),
        long=sum(1 for v in vals if v > 0),
        short=sum(1 for v in vals if v < 0),
        neutral=sum(1 for v in vals if v == 0),
    )


def cost_cell(bars, signal_fn, lookback, cfg):
    """One (signal, lookback, cost) walk-forward regime-stability cell."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    res = bt.stress_segments(
        signals_fn=signal_fn,
        bars=bars,
        signals=signal_fn(closes, lookback=lookback),
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    medians = [round(s.baseline_median_log_return, 6) for s in res.scenarios]
    nulls = [round(s.noise_median_log_return, 6) for s in res.scenarios]
    positive_edge = (res.overall_verdict == "REGIME_STABLE"
                     and all(m > 0 for m in medians))
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=medians,
        null_medians=nulls,
        edge_status=[int(s.edge_status) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 6),
        null_dispersion=round(res.null_dispersion, 6),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
        positive_edge=bool(positive_edge),
        above_null=bool(all(m > n for m, n in zip(medians, nulls))),
        mean_candidate_median=round(float(np.mean(medians)), 6),
        mean_null_median=round(float(np.mean(nulls)), 6),
    )


def classify_cost_robustness(cells):
    """Classify one (signal, lookback, ticker) across the cost grid.

    NO_EDGE      : no uniformly-positive REGIME_STABLE edge even at zero cost.
    COST_ROBUST  : positive edge at zero, realistic AND conservative cost.
    COST_FRAGILE : positive edge at zero cost but lost at realistic or
                   conservative cost.
    """
    if not cells["zero"]["positive_edge"]:
        return "NO_EDGE"
    if (cells["realistic"]["positive_edge"]
            and cells["conservative"]["positive_edge"]):
        return "COST_ROBUST"
    return "COST_FRAGILE"


def aggregate(cells_by_ticker, ticker_order):
    counts = {"NO_EDGE": 0, "COST_ROBUST": 0, "COST_FRAGILE": 0}
    per_asset = {}
    for t in ticker_order:
        cls = classify_cost_robustness(cells_by_ticker[t])
        counts[cls] += 1
        per_asset[t] = dict(
            classification=cls,
            verdicts={lvl: cells_by_ticker[t][lvl]["verdict"] for lvl in COST_ORDER},
            mean_candidate_median={lvl: cells_by_ticker[t][lvl]["mean_candidate_median"]
                                   for lvl in COST_ORDER},
        )
    mean_medians = {
        lvl: round(float(np.mean([cells_by_ticker[t][lvl]["mean_candidate_median"]
                                  for t in ticker_order])), 6)
        for lvl in COST_ORDER
    }
    # Decision rule on the assets with a zero-cost positive edge.
    edge_assets = [t for t in ticker_order if cells_by_ticker[t]["zero"]["positive_edge"]]
    robust = [t for t in edge_assets if per_asset[t]["classification"] == "COST_ROBUST"]
    if not edge_assets:
        decision = "NO_EDGE_AT_ZERO"
    elif len(robust) == len(edge_assets):
        decision = "SURVIVES_CONSERVATIVE"
    elif robust:
        decision = "PARTIALLY_SURVIVES"
    else:
        decision = "DEGRADES"
    return dict(
        counts=counts,
        mean_candidate_median_by_cost=mean_medians,
        zero_cost_edge_assets=edge_assets,
        n_zero_cost_edge=len(edge_assets),
        n_cost_robust=len(robust),
        cost_robust_assets=robust,
        decision=decision,
        per_asset=per_asset,
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: "
              f"{'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        return 1

    # Load universe.
    all_tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}
    ticker_order = list(all_tickers.keys())

    print("\n=== 3. Signal-correctness diagnostic (existing vs corrected) ===")
    print("  Weight composition after the warmup window, lookback=5:")
    print(f"  {'ticker':6s} {'existing long/short/neutral':>30s} {'corrected long/short/neutral':>32s} {'existing==always_long':>22s}")
    signal_diagnostic = {}
    for t in ticker_order:
        closes = all_tickers[t].closes_array()
        ex = weight_counts(bt.momentum_signals(closes, 5), 5)
        co = weight_counts(corrected_momentum_signals(closes, 5), 5)
        ex_sig = bt.momentum_signals(closes, 5)
        al = bt.always_long_signal(closes, 5, 5)
        is_always_long = all(a.weight == b.weight for a, b in zip(ex_sig[5:], al[5:]))
        signal_diagnostic[t] = dict(
            existing=ex, corrected=co, existing_is_always_long=is_always_long)
        print(f"  {t:6s} "
              f"{str(ex['long'])+'/'+str(ex['short'])+'/'+str(ex['neutral']):>30s} "
              f"{str(co['long'])+'/'+str(co['short'])+'/'+str(co['neutral']):>32s} "
              f"{str(is_always_long):>22s}")
    n_always_long = sum(1 for t in ticker_order if signal_diagnostic[t]["existing_is_always_long"])
    print(f"  existing == always_long for {n_always_long}/{len(ticker_order)} tickers "
          f"(mean-log-price defect -> always long).")

    print("\n=== 4. Leakage review (existing + corrected, lookback=5, AMZN/JPM) ===")
    for t in ("AMZN", "JPM"):
        bars = all_tickers[t]
        for vname, fn in SIGNAL_VARIANTS.items():
            sig = fn(bars.closes_array(), lookback=5)
            bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
            res = bt.run_bars(list(bars), sig, bt.BacktestConfig(initial_capital=1e6))
            bt.check_equity_matches_fills(res.equity_curve, res.trades, bars.closes_array(), 1e6)
        print(f"  {t}: both variants pass signal-integrity + fill-equity audit [PASS]")

    print("\n=== 5. Cost grid sweep (signal x lookback x cost x universe) ===")
    print(f"  cost levels: {COST_ORDER}")
    for lvl in COST_ORDER:
        c = COST_LEVELS[lvl]
        print(f"    {lvl:12s} comm/trade={c['commission_per_trade']} "
              f"comm/share={c['commission_per_share']} "
              f"slip_cents={c['slippage_cents']} "
              f"slip_prop={c['slippage_proportional']}")

    cells = {v: {lb: {lvl: {} for lvl in COST_ORDER} for lb in LOOKBACKS}
             for v in SIGNAL_VARIANTS}
    for v in SIGNAL_VARIANTS:
        for lb in LOOKBACKS:
            for lvl in COST_ORDER:
                cfg = cost_config(lvl)
                for t in ticker_order:
                    cells[v][lb][lvl][t] = cost_cell(all_tickers[t], SIGNAL_VARIANTS[v], lb, cfg)
        print(f"  variant={v}: completed lookbacks {LOOKBACKS} x costs {COST_ORDER}")

    print("\n=== 6. Cost-robustness classification ===")
    classification = {}
    for v in SIGNAL_VARIANTS:
        classification[v] = {}
        for lb in LOOKBACKS:
            cells_by_ticker = {t: {lvl: cells[v][lb][lvl][t] for lvl in COST_ORDER}
                               for t in ticker_order}
            classification[v][str(lb)] = aggregate(cells_by_ticker, ticker_order)

    for v in SIGNAL_VARIANTS:
        print(f"\n  variant={v}")
        for lb in LOOKBACKS:
            a = classification[v][str(lb)]
            mm = a["mean_candidate_median_by_cost"]
            print(f"    lookback={lb:2d}: counts={a['counts']} "
                  f"mean_median zero/real/cons="
                  f"{mm['zero']:+.3f}/{mm['realistic']:+.3f}/{mm['conservative']:+.3f} "
                  f"decision={a['decision']}")
            if lb == REFERENCE_LOOKBACK:
                for t in ticker_order:
                    pa = a["per_asset"][t]
                    print(f"        {t:6s} {pa['classification']:12s} "
                          f"medians {pa['mean_candidate_median']}")

    # Cross-consistency: existing/zero/lookback=5 must reproduce the prior sweep.
    print("\n=== 7. Cross-consistency vs prior momentum_lookback_sweep (existing, zero, lb=5) ===")
    cross = dict(check=True, mismatches=[])
    if PRIOR_SWEEP_PATH.exists():
        with open(PRIOR_SWEEP_PATH) as fh:
            prior = json.load(fh)
        for t in ticker_order:
            prior_med = prior["per_asset"][t]["5"]["medians"]
            now_med = cells["existing"][REFERENCE_LOOKBACK]["zero"][t]["medians"]
            ok = len(prior_med) == len(now_med) and all(
                abs(a - b) <= 1e-3 for a, b in zip(prior_med, now_med))
            cross["mismatches"].append(dict(ticker=t, prior=prior_med, now=now_med, match=ok))
            print(f"  {t:6s} prior={prior_med} now={now_med} match={ok}")
            cross["check"] = cross["check"] and ok
        print(f"  CROSS_CONSISTENCY: {cross['check']}")
    else:
        print("  prior sweep artifact not found; cross-consistency skipped")
        cross = dict(check=None, mismatches=[])

    # Determinism: rerun a target subset and compare.
    print("\n=== 8. Determinism (target subset, both variants, realistic) ===")
    target = ["AMZN", "JPM"]
    rerun = {}
    for v in SIGNAL_VARIANTS:
        for lb in (3, 5, 10):
            for lvl in COST_ORDER:
                rerun[(v, lb, lvl)] = {
                    t: cost_cell(all_tickers[t], SIGNAL_VARIANTS[v], lb, cost_config(lvl))
                    for t in target}
    det_ok = True
    for key, val in rerun.items():
        v, lb, lvl = key
        for t in target:
            if json.dumps(val[t], sort_keys=True) != json.dumps(cells[v][lb][lvl][t], sort_keys=True):
                det_ok = False
                print(f"  NON-DETERMINISTIC: {v} lb={lb} {lvl} {t}")
    print(f"  determinism r1 == r2: {det_ok}")
    assert det_ok, "non-deterministic cost-sensitivity output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        reference_lookback=REFERENCE_LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_levels={lvl: COST_LEVELS[lvl] for lvl in COST_ORDER},
        signal_variants=list(SIGNAL_VARIANTS.keys()),
        signal_diagnostic=signal_diagnostic,
        signal_diagnostic_summary=dict(
            existing_is_always_long_count=n_always_long,
            n_tickers=len(ticker_order),
            note=("momentum_signals computes mean log PRICE, not mean log RETURN; "
                  "for adjusted closes > 1.0 the weight is constant +1 (always long)."),
        ),
        cells=cells,
        classification=classification,
        cross_consistency_vs_prior=cross,
        determinism=bool(det_ok),
    )
    with open(ARTIFACT_PATH, "w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Cost-sensitivity classification is exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())