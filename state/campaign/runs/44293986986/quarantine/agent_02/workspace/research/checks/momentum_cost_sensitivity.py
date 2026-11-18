"""Real-data transaction-cost sensitivity check for short-horizon momentum.

Objective (campaign task R-001): the short-horizon momentum signal
(`bt.momentum_signals`, lookbacks 3 / 5 / 10) is admitted as *candidate positive
evidence* in `state/LEARNING_STATE.md` on the strength of zero-cost walk-forward
medians (activations 37318950814 and 37361000967). The binding unresolved
qualification is execution-cost robustness. This check asks, a-priori and on the
collected 10-asset adjusted-close universe:

    Does the short-horizon momentum edge survive conservative transaction-cost
    stress, relative to a cost-matched coin-flip null?

A-priori method (fixed before any cost result is seen; not tuned to OOS):

- Universe: the 10 collected tickers (manifest `yf-ohlcv-universe-2009-to-2026-10-03`).
- Signal: `bt.momentum_signals(closes, lookback)` for lookback in {3, 5, 10};
  long the previous N-day return, hold 1 day, daily rebalance. Unchanged from the
  candidate-evidence check; no rule tuning.
- Segments: 4 contiguous volatility blocks (`bt.volatility_blocks`, window=60),
  identical to `research/checks/momentum_lookback_sweep.py`.
- Walk-forward: train=252 / test=84 / warmup=60 / overlap=60, min 400 bars/segment.
- Cost model: the repository's existing realistic cost configuration, taken
  verbatim from `examples/ma_crossover_real_data.py` and `examples/ma_crossover.py`:

      commission_per_trade=2.0, commission_per_share=0.003,
      slippage_cents=2.0, slippage_proportional=0.0005

  scaled by a predeclared cost multiplier grid

      COST_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0, 3.0)

  The cost config is passed to BOTH the candidate (via `stress_segments` ->
  `parameter_sweep`) and the matched coin-flip null (via `stress_segments` ->
  `noise_benchmark`), so the null is cost-matched to the candidate at every
  level. Because `noise_benchmark` derives its coin-flip seed deterministically
  from the lookback parameter alone, the null draw is identical across cost
  levels; only the cost differs, isolating the cost effect.

- Zero-cost column (multiplier 0.0) must reproduce the recorded candidate-
  evidence artifact `momentum_lookback_sweep_results.json` exactly (medians,
  null medians, verdicts) — a cross-consistency gate tying this check to the
  existing evidence.

Predeclared classification (per asset x lookback), computed from the candidate's
absolute aggregate level

    agg_cand(m) = median over segments of the candidate segment median log return
                  at cost multiplier m

with REALISTIC_COST = 1.0 and ROBUST_COST = 2.0:

    COST_ROBUST  : agg_cand(1.0) > 0 AND agg_cand(2.0) > 0   (edge survives 2x stress)
    COST_FRAGILE : agg_cand(1.0) > 0 AND agg_cand(2.0) <= 0  (edge at realistic only)
    NO_EDGE      : agg_cand(1.0) <= 0                         (no edge under realistic)

A secondary matched-null classification on gap(m) = agg_cand(m) - agg_null(m) is
reported as VS_NULL_ROBUST / VS_NULL_FRAGILE / VS_NULL_NONE for corroboration.
Both classifications are derived from the rounded per-segment medians stored in
the artifact, so they are exactly reproducible from the artifact.

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

# Existing realistic cost model (examples/ma_crossover_real_data.py,
# examples/ma_crossover.py) and the predeclared cost-multiplier grid.
BASE_REALISTIC = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
COST_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0, 3.0)
REALISTIC_COST = 1.0
ROBUST_COST = 2.0

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def cost_config(multiplier):
    """BacktestConfig for a cost multiplier of the existing realistic model."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=BASE_REALISTIC["commission_per_trade"] * multiplier,
        commission_per_share=BASE_REALISTIC["commission_per_share"] * multiplier,
        slippage_cents=BASE_REALISTIC["slippage_cents"] * multiplier,
        slippage_proportional=BASE_REALISTIC["slippage_proportional"] * multiplier,
    )


def segment_result(bars, lookback, multiplier):
    """Walk-forward regime-stability result for one ticker/lookback/cost level.

    Mirrors `research/checks/momentum_lookback_sweep.py::segment_result`, but
    threads a cost `cfg` into `stress_segments` so that both the candidate and
    the matched coin-flip null are charged the same transaction costs.
    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError("signals {} != bars {}".format(len(sig), len(closes)))
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
    )


def classify(agg_cand, agg_null):
    """Predeclared cost-robustness and matched-null classifications.

    `agg_cand` / `agg_null` map cost-multiplier (str) -> aggregate level.
    Levels are already rounded (median of rounded per-segment medians).
    """
    def at(m):
        return agg_cand[repr(float(m))]

    survives_1 = at(REALISTIC_COST) > 0.0
    survives_2 = at(ROBUST_COST) > 0.0
    if survives_1 and survives_2:
        cost_rob = "COST_ROBUST"
    elif survives_1 and not survives_2:
        cost_rob = "COST_FRAGILE"
    else:
        cost_rob = "NO_EDGE"

    def gap(m):
        return agg_cand[repr(float(m))] - agg_null[repr(float(m))]

    gap_1 = gap(REALISTIC_COST)
    gap_2 = gap(ROBUST_COST)
    if gap_1 > 0.0 and gap_2 > 0.0:
        vs_null = "VS_NULL_ROBUST"
    elif gap_1 > 0.0 and gap_2 <= 0.0:
        vs_null = "VS_NULL_FRAGILE"
    else:
        vs_null = "VS_NULL_NONE"

    survives_to = 0.0
    for m in COST_MULTIPLIERS:
        if agg_cand[repr(float(m))] > 0.0:
            survives_to = float(m)
    return cost_rob, vs_null, survives_to


def signal_diagnostic(bars, lookback):
    """Diagnostic of the actual signal series produced by `bt.momentum_signals`.

    Reports the set of nonzero weights, the number of sign flips, and the range
    of the thresholded quantity `mean(log(closes[i-lookback+1:i+1]))` (the mean
    of LOG PRICES, per the framework implementation). A genuine N-day momentum
    signal must take both signs and flip; a constant-weight series is a static
    directional bet, not a momentum signal.

    This is a diagnostic of the candidate signal under test; it introduces no
    new signal rule and does not alter any result.
    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    sig = bt.momentum_signals(closes, lookback=lookback)
    weights = np.array([s.weight for s in sig], dtype=np.float64)
    nonzero = weights[weights != 0.0]
    flips = int(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
    meanlog = np.array(
        [float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
         for i in range(lookback, len(closes))], dtype=np.float64)
    return dict(
        weight_set=sorted(set(float(w) for w in nonzero)),
        n_nonzero=int(len(nonzero)),
        sign_flips=flips,
        min_mean_log_price=round(float(meanlog.min()), 4),
        max_mean_log_price=round(float(meanlog.max()), 4),
        degenerate=bool(len(set(float(w) for w in nonzero)) <= 1),
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print("  {}: {}".format(entry["ticker"], status))
    print("  dataset: {} | collection: {} | universe: {}".format(
        manifest["dataset_id"], manifest["collection_date"], len(manifest["universe"])))

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
        signals = bt.momentum_signals(closes, lookback=CANONICAL_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        # Cost-aware equity-fill audit at the realistic (1x) cost: every fill,
        # including commission, must reconcile with the marked equity curve.
        res = bt.run_bars(list(bars), signals, cost_config(REALISTIC_COST))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print("  {}: {} bars, momentum signals {}, leakage [PASS], "
              "costed(1x) total_return {:+.3f}".format(
                  ticker, bars.n_bars, len(signals), total_return))
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; the "
          "cost path reconciles fills+commission with the equity curve.")

    print("\n=== 4. Cost-sensitivity sweep (lookbacks {} x cost {} x universe) ==="
          .format(list(LOOKBACKS), list(COST_MULTIPLIERS)))
    print("  Base realistic cost model: {}".format(BASE_REALISTIC))
    print("  Cost multipliers: {} (x base realistic)".format(list(COST_MULTIPLIERS)))
    print("  Walk-forward train={}/test={}/warmup={}/overlap={}, {} vol blocks, "
          "matched coin-flip null per cost level\n".format(
              TRAIN, TEST, WARM, OVERLAP, N_BLOCKS))

    all_tickers = {}
    for entry in manifest["entries"]:
        all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(all_tickers.keys())

    mult_keys = [repr(float(m)) for m in COST_MULTIPLIERS]

    per_asset = {}
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        per_asset[ticker] = {}
        for lb in LOOKBACKS:
            cost_levels = {}
            for m in COST_MULTIPLIERS:
                cost_levels[repr(float(m))] = segment_result(bars, lb, m)
            # Aggregate levels from the rounded per-segment medians.
            agg_cand = {}
            agg_null = {}
            gap = {}
            for mk in mult_keys:
                cm = cost_levels[mk]["medians"]
                nm = cost_levels[mk]["null_medians"]
                agg_cand[mk] = round(float(np.median(cm)), 3) if cm else 0.0
                agg_null[mk] = round(float(np.median(nm)), 3) if nm else 0.0
                gap[mk] = round(agg_cand[mk] - agg_null[mk], 3)
            cost_rob, vs_null, survives_to = classify(agg_cand, agg_null)
            per_asset[ticker][str(lb)] = dict(
                cost_levels=cost_levels,
                agg_cand=agg_cand,
                agg_null=agg_null,
                gap=gap,
                cost_robustness=cost_rob,
                vs_null_robustness=vs_null,
                survives_to_multiplier=survives_to,
            )

    # Universe-level tallies.
    def tally(key):
        by_lb = {}
        for lb in LOOKBACKS:
            counts = {}
            for t in ticker_order:
                v = per_asset[t][str(lb)][key]
                counts[v] = counts.get(v, 0) + 1
            by_lb[str(lb)] = counts
        all_counts = {}
        for lb in LOOKBACKS:
            for k, v in by_lb[str(lb)].items():
                all_counts[k] = all_counts.get(k, 0) + v
        by_lb["all"] = all_counts
        return by_lb

    cost_counts = tally("cost_robustness")
    vs_null_counts = tally("vs_null_robustness")

    print("  Cost-robustness classification counts by lookback:")
    for lb in list(LOOKBACKS) + ["all"]:
        print("    lookback {:>2}: {}".format(lb, cost_counts[str(lb)]))
    print("\n  Matched-null (candidate - null) classification counts by lookback:")
    for lb in list(LOOKBACKS) + ["all"]:
        print("    lookback {:>2}: {}".format(lb, vs_null_counts[str(lb)]))

    print("\n  Per-asset candidate agg_cand(m) curves and cost verdicts:")
    for ticker in ticker_order:
        parts = []
        for lb in LOOKBACKS:
            p = per_asset[ticker][str(lb)]
            curve = ",".join(
                "{:+.3f}".format(p["agg_cand"][mk]) for mk in mult_keys)
            parts.append("lb{}[{}] {}".format(lb, curve, p["cost_robustness"]))
        print("    {:6s} {}".format(ticker, " | ".join(parts)))
    print("")

    # Zero-cost cross-consistency gate vs the recorded momentum_lookback_sweep
    # artifact: at multiplier 0.0 the medians, null medians and verdicts must
    # reproduce the candidate-evidence artifact exactly.
    print("=== 5. Zero-cost column cross-check vs momentum_lookback_sweep_results.json ===")
    with open(SWEEP_ARTIFACT) as f:
        sweep = json.load(f)
    cmismatch = False
    for ticker in ticker_order:
        for lb in LOOKBACKS:
            pub = sweep["per_asset"][ticker][str(lb)]
            got = per_asset[ticker][str(lb)]["cost_levels"][repr(0.0)]
            ok = (got["medians"] == pub["medians"]
                  and got["null_medians"] == pub["null_medians"]
                  and got["verdict"] == pub["verdict"])
            if not ok:
                cmismatch = True
                print("  MISMATCH {} lb{}: medians {} vs {} | nulls {} vs {} | "
                      "verdict {} vs {}".format(
                          ticker, lb, got["medians"], pub["medians"],
                          got["null_medians"], pub["null_medians"],
                          got["verdict"], pub["verdict"]))
    if cmismatch:
        print("ZERO-COST CROSS-CHECK FAILED; do not record results.")
        sys.exit(1)
    print("  all {} asset x lookback cells MATCH the recorded zero-cost "
          "artifact.".format(len(ticker_order) * len(LOOKBACKS)))

    # Determinism: rerun a subset of (asset, lookback, cost) and assert
    # identical output.
    print("\n=== 6. Determinism (AMZN lb5 all costs, JPM lb3 all costs) ===")
    det_ok = True
    for ticker, lb in [("AMZN", 5), ("JPM", 3)]:
        bars = all_tickers[ticker]
        rerun = {repr(float(m)): segment_result(bars, lb, m) for m in COST_MULTIPLIERS}
        orig = per_asset[ticker][str(lb)]["cost_levels"]
        same = json.dumps(rerun, sort_keys=True) == json.dumps(orig, sort_keys=True)
        det_ok = det_ok and same
        print("  {} lb{}: r1 == r2 -> {}".format(ticker, lb, same))
    assert det_ok, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        method=dict(
            lookbacks=list(LOOKBACKS),
            cost_multipliers=list(COST_MULTIPLIERS),
            base_realistic=BASE_REALISTIC,
            realistic_cost=REALISTIC_COST,
            robust_cost=ROBUST_COST,
            canonical_lookback=CANONICAL_LOOKBACK,
            train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
            window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
            classification_rule=(
                "COST_ROBUST if agg_cand(1.0)>0 and agg_cand(2.0)>0; "
                "COST_FRAGILE if agg_cand(1.0)>0 and agg_cand(2.0)<=0; "
                "NO_EDGE if agg_cand(1.0)<=0. agg_cand(m)=median over segments "
                "of candidate segment median log return at multiplier m."),
        ),
        cost_grid=[dict(
            multiplier=float(m),
            commission_per_trade=BASE_REALISTIC["commission_per_trade"] * m,
            commission_per_share=BASE_REALISTIC["commission_per_share"] * m,
            slippage_cents=BASE_REALISTIC["slippage_cents"] * m,
            slippage_proportional=BASE_REALISTIC["slippage_proportional"] * m,
        ) for m in COST_MULTIPLIERS],
        per_asset=per_asset,
        universe=dict(
            cost_robustness_counts_by_lookback=cost_counts,
            vs_null_counts_by_lookback=vs_null_counts,
            ticker_order=ticker_order,
            canonical_lookback_5=dict(
                cost_robustness_counts=cost_counts[str(CANONICAL_LOOKBACK)],
                vs_null_counts=vs_null_counts[str(CANONICAL_LOOKBACK)],
                per_asset=dict(
                    (t, dict(
                        cost_robustness=per_asset[t][str(CANONICAL_LOOKBACK)]["cost_robustness"],
                        vs_null_robustness=per_asset[t][str(CANONICAL_LOOKBACK)]["vs_null_robustness"],
                        survives_to_multiplier=per_asset[t][str(CANONICAL_LOOKBACK)]["survives_to_multiplier"],
                        agg_cand=per_asset[t][str(CANONICAL_LOOKBACK)]["agg_cand"],
                        agg_null=per_asset[t][str(CANONICAL_LOOKBACK)]["agg_null"],
                    ))
                    for t in ticker_order
                ),
            ),
        ),
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