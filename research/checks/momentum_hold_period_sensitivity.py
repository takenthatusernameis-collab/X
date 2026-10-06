"""Holding-period sensitivity of the qualified lookback-5 momentum signal.

This is the R-002 focused task. The momentum class (long the previous
lookback-day return, hold 1 day, daily rebalance) has been admitted as
candidate positive evidence: REGIME_STABLE positive edge in 7/10 of the
collected universe at a one-day hold. This test varies only the holding
period -- not the lookback -- to answer the primary question:

  Does the qualified momentum effect survive a small predeclared
  holding-period variation without becoming a single-point timing
  artifact?

Method (fixed a-priori, not tuned to OOS): lookback fixed at 5. Holding
periods are predeclared as HOLD_PERIODS = (1, 2, 3, 5): at a rebalance bar
the position is set to the signal and carried for HOLD days before the next
rebalance (signal at bar t is held during bars t .. t+H-1; a fresh
rebalance occurs at bars lookback + k*H for k = 0, 1, 2, ...). The same
walk-forward windows, 4 contiguous volatility blocks, min_segment_bars
threshold, and coin-flip null as the existing momentum check
(`research/checks/momentum.py`) are used; the only change is a hold-period
transform applied to the signal series before the walk-forward. The null is
the framework's coin-flip benchmark run on the same segments -- identical to
momentum.py -- so the H=1 run is a direct replication of the admitted
evidence.

Classification scheme (predeclared; applied after the grid runs, not used
to select parameters):
  - Per asset per hold: verdict per segment + per-asset verdict
    (REGIME_STABLE positive / CONSISTENT_WITH_NOISE / REGIME_DEPENDENT /
     REGIME_STABLE_LOSS).
  - Hold profile per asset:
      STABLE_EDGE     = REGIME_STABLE with uniformly positive medians at
                        every hold in HOLD_PERIODS;
      WEAKENS         = positive REGIME_STABLE at some holds,
                        CONSISTENT_WITH_NOISE or regime-dependent at others;
      SINGULAR        = a positive edge at exactly one hold period
                        (anywhere in the grid);
      NO_EDGE         = no positive edge at any hold.
  - Universe-level classification of the effect:
      HORIZON_STABLE        = the 7/10 H=1 REGIME_STABLE-positive assets
                              remain uniformly positive REGIME_STABLE at
                              all holds in the grid; no asset shows a
                              SINGULAR hold profile; the effect is not a
                              single-point timing artifact.
      HORIZON_SENSITIVE     = the edge's presence depends on hold period:
                              present at some holds, absent or weaker at
                              others, but not confined to a single hold.
      HORIZON_SINGULAR      = the positive edge is confined to exactly one
                              hold period across the universe -- a
                              single-point timing artifact.
      HORIZON_DESTROYED     = no positive edge survives at any hold.

A-priori falsification prediction: if the H=1 edge is a timing artifact of
daily rebalancing, extending the hold should erase it: medians should
decline with hold period and the REGIME_STABLE-positive set should collapse
to zero or become confined to a single hold (HORIZON_SINGULAR /
HORIZON_DESTROYED). If the edge is a durable short-horizon autocorrelation,
it should persist with roughly stable sign and magnitude across the small
grid (HORIZON_STABLE), with perhaps modest decay. Either outcome closes the
cell.

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
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
HOLD_PERIODS = (1, 2, 3, 5)  # predeclared hold-period grid; tuned after results is out of scope.
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_hold_period_sensitivity_results.json"


def hold_effective_signals(signals, hold):
    """Carry each signal for `hold` bars before the next rebalance.

    Position at bar b equals the signal from the most recent rebalance bar
    b' <= b, where rebalance bars are lookback + k*hold. Concretely,
    effective[b] = signals[b - (b - lookback) % hold] for b >= lookback,
    else neutral. This is the standard holding-period semantics: the bet
    taken at bar t is marked to market for bars t .. t+hold-1 and exited
    at bar t+hold (or re-entered at the new level).

    No look-ahead: the transform only reindexes the signal series; each
    output bar depends on signals at or before that bar.
    """
    n = len(signals)
    out = list(signals)
    for i in range(n):
        if i < LOOKBACK:
            out[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            idx = i - (i - LOOKBACK) % hold
            out[i] = bt.Signal(date=i + 1, weight=signals[idx].weight)
    return out


def run_asset(asset, tickers, hold):
    """Regime-stability gate for one asset at one hold period."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=LOOKBACK)
    eff = hold_effective_signals(sig, hold)
    if len(eff) != len(closes):
        raise ValueError(f"{asset} hold={hold}: effective signals {len(eff)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=lambda c, **p: hold_effective_signals(bt.momentum_signals(c, lookback=LOOKBACK), hold),
        bars=list(bars),
        signals=eff,
        param_grid=[{"hold": hold}],
        baseline=(("hold", hold),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    return res


def perturbation_sweep(n_bars=600):
    """Synthetic perturbation sweep over the predeclared hold grid on the
    canonical regime family, with a coin-flip null benchmark."""
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = [{"hold": float(h)} for h in HOLD_PERIODS]
    sweep = bt.parameter_sweep(
        signals_fn=lambda c, **p: hold_effective_signals(
            bt.momentum_signals(c, lookback=LOOKBACK), int(round(p["hold"]))),
        bars=list(bars),
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    summary = bt.sweep_summary(sweep, baseline=(("hold", 1.0),))
    noise = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )
    return dict(
        param_sets=[tuple(ps) for ps in sweep.param_sets],
        medians=[round(m, 3) for m in summary.median_log_returns],
        baseline_median=round(summary.baseline_median_log_return, 3),
        noise_median=round(noise.baseline_median_log_return, 3),
        compare_noise_default=summary.compare_noise(noise.baseline_median_log_return),
        compare_noise_fixed=summary.compare_noise(noise.baseline_median_log_return, tol=0.05),
        effective_tolerance=round(summary.effective_tolerance, 4),
        n_folds=summary.n_folds,
    )


def asset_hold_profile(medians):
    """Classify one asset's edge across the hold grid (per-asset medians,
    first segment median is the canonical cross-segment edge proxy)."""
    has_pos = [m > 0 for m in medians]
    if all(has_pos):
        return "STABLE_EDGE"
    if any(has_pos):
        n_pos = sum(has_pos)
        if n_pos == 1:
            return "SINGULAR"
        return "WEAKENS"
    return "NO_EDGE"


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

    print("\n=== 3. Leakage review per asset, per hold ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        for hold in HOLD_PERIODS:
            eff = hold_effective_signals(bt.momentum_signals(closes, lookback=LOOKBACK), hold)
            bt.check_signal_integrity(eff, [b.date for b in bars], warmup=0)
            cfg = bt.BacktestConfig(initial_capital=1e6)
            res = bt.run_bars(list(bars), eff, cfg)
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
            turns = sum(1 for i in range(1, len(eff)) if eff[i].weight != eff[i - 1].weight)
            print(f"  {ticker} hold={hold}: {bars.n_bars} bars, "
                  f"effective signals {len(eff)}, leakage [PASS], "
                  f"turnover {turns}x | total_return {(res.equity_curve[-1] - 1e6) / 1e6:+.3f}")
    print("  no-look-ahead: effective signal at bar t reindexes the momentum "
          "sign at bars <= t only; neutral bars before the lookback window "
          "carry the equity forward unchanged.")

    print("\n=== 4. Holding-period sensitivity on REGIME_DEPENDENT assets ===")
    print("Momentum: long the previous {}-day return; hold = {} days per "
          "predeclared grid. Params: walk-forward per segment "
          "train={}d / test={}d / warmup={}d / overlap={}d, min {} "
          "bars/segment, 4 contiguous volatility blocks, compared vs coin-flip "
          "null on the same segments\n".format(
          LOOKBACK, " / ".join(str(h) for h in HOLD_PERIODS), TRAIN, TEST, WARM,
          OVERLAP, MIN_SEGMENT_BARS))
    print("Segments: {} contiguous blocks by date, past-only volatility "
          "classifier (turbulent = block median trailing-{}-bar realized vol "
          "> series-wide median)\n".format(N_BLOCKS, WINDOW))

    per_asset = {}
    for asset in target:
        holds = {}
        for hold in HOLD_PERIODS:
            res = run_asset(asset, tickers, hold)
            medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
            holds[str(hold)] = dict(
                verdict=res.overall_verdict,
                segments=[s.name for s in res.scenarios],
                medians=medians,
                null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
                candidate_dispersion=round(res.candidate_dispersion, 3),
                null_dispersion=round(res.null_dispersion, 3),
                n_folds=res.n_folds,
            )
            print("  {} hold={}: verdict {:18s} cand_disp {:+.3f} null_disp {:+.3f} "
                  "cand_medians {}".format(
                asset, hold, res.overall_verdict, res.candidate_dispersion,
                res.null_dispersion, str(medians)))
        per_asset[asset] = holds

    # Full-universe generalization: hold-period grid on all collected tickers.
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    hold_sums = {}
    for hold in HOLD_PERIODS:
        summary = bt.stress_segments_across_tickers(
            tickers=tickers_univ,
            signals_fn=lambda c, h=hold, **kw: hold_effective_signals(
                bt.momentum_signals(c, lookback=LOOKBACK), h),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("hold", float(hold)),),
            param_grid=[{"hold": hold}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        hold_sums[str(hold)] = dict(
            verdict_counts=summary.verdict_counts,
            ticker_order=summary.ticker_order,
            per_asset=dict(
                (t, dict(
                    segments=[s.name for s in r.scenarios],
                    medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
                    null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
                    candidate_dispersion=round(r.candidate_dispersion, 3),
                    null_dispersion=round(r.null_dispersion, 3),
                    verdict=r.overall_verdict,
                ))
                for t, r in summary.assets.items()
            ),
        )
        print("=== Hold {} across collected universe ===".format(hold))
        print("  " + summary.inspect().replace("\n", "\n  "))
        print("")

    # Universe-level classification (predeclared scheme).
    print("=== 5. Universe-level holding-period classification ===")
    ticker_order = hold_sums[str(HOLD_PERIODS[0])]["ticker_order"]
    edge_counts = {str(h): 0 for h in HOLD_PERIODS}
    singular_assets = []
    stable_assets = []
    profiles = {}
    for t in ticker_order:
        asset_profiles = {}
        for h in HOLD_PERIODS:
            prof = hold_sums[str(h)]["per_asset"][t]
            prof = {k: v for k, v in prof.items() if k != "segments"}
            asset_profiles[str(h)] = prof
            medians = prof["medians"]
            is_edge = prof["verdict"] == "REGIME_STABLE" and all(m > 0 for m in medians)
            if is_edge:
                edge_counts[str(h)] += 1
        profiles[t] = asset_profiles
    for t in ticker_order:
        p = profiles[t]
        medians = [p[str(h)]["medians"] for h in HOLD_PERIODS]
        first_seg_medians = [mm[0] for mm in medians]
        # Positive edge if the canonical first-segment median is positive and
        # the overall verdict is REGIME_STABLE at that hold.
        has_pos = []
        for h in HOLD_PERIODS:
            prof = p[str(h)]
            is_edge = prof["verdict"] == "REGIME_STABLE" and all(m > 0 for m in prof["medians"])
            has_pos.append(is_edge)
        n_pos = sum(has_pos)
        if all(has_pos):
            profile = "STABLE_EDGE"
            stable_assets.append(t)
        elif n_pos == 1:
            profile = "SINGULAR"
            singular_assets.append(t)
        elif n_pos > 1:
            profile = "WEAKENS"
        else:
            profile = "NO_EDGE"
        print("  {:6s}: edge_at_holds={} -> {}".format(t, str(has_pos), profile))

    # Universe-level classification.
    n_edge_at_h1 = edge_counts[str(HOLD_PERIODS[0])]
    n_edge_all = min(edge_counts.values())
    if n_edge_all >= 7 and len(singular_assets) == 0 and n_edge_at_h1 >= 7:
        effect_class = "HORIZON_STABLE"
    elif len(singular_assets) > 0:
        effect_class = "HORIZON_SINGULAR"
    elif n_edge_at_h1 == 0:
        effect_class = "HORIZON_DESTROYED"
    else:
        effect_class = "HORIZON_SENSITIVE"
    print("  H=1 positive edges: {} | min across grid: {} | SINGULAR assets: {} "
          "-> {}".format(n_edge_at_h1, n_edge_all, len(singular_assets), effect_class))
    print("  stable edge across all holds ({}): {}".format(n_edge_all,
          ", ".join(stable_assets) if stable_assets else "none"))
    print("")

    # Synthetic perturbation sweep over the hold grid.
    pers = perturbation_sweep()
    print("=== 6. Synthetic perturbation sweep (hold 1 / 2 / 3 / 5) ===")
    print("  param sets: {}".format(pers["param_sets"]))
    print("  candidate medians: {}".format(pers["medians"]))
    print("  baseline (hold=1): {:+.3f}".format(pers["baseline_median"]))
    print("  coin-flip null baseline: {:+.3f}".format(pers["noise_median"]))
    print("  baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(pers["effective_tolerance"], pers["compare_noise_default"]))
    print("  baseline vs null (fixed tol 0.05): {}"
          .format(pers["compare_noise_fixed"]))
    print("")

    # Determinism: rerun target subset and compare the normalized
    # serialized representation already captured in per_asset. The first
    # attempt incorrectly treated those persisted dict records as framework
    # result objects, which caused AttributeError during the determinism gate.
    res2 = {a: {h: run_asset(a, tickers, h) for h in HOLD_PERIODS} for a in target}
    out1 = json.dumps({a: {str(h): dict(
            verdict=r["verdict"],
            medians=r["medians"],
            null_medians=r["null_medians"],
            candidate_dispersion=r["candidate_dispersion"],
            null_dispersion=r["null_dispersion"],
            n_folds=r["n_folds"],
        ) for h, r in hres.items()} for a, hres in per_asset.items()}, sort_keys=True)
    out2 = json.dumps({a: {str(h): dict(
            verdict=r.overall_verdict,
            medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
            null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
            candidate_dispersion=round(r.candidate_dispersion, 3),
            null_dispersion=round(r.null_dispersion, 3),
            n_folds=r.n_folds,
        ) for h, r in hres.items()} for a, hres in res2.items()}, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Assemble artifact.
    per_asset_out = {}
    for asset in target:
        per_asset_out[asset] = {}
        for h in HOLD_PERIODS:
            res = run_asset(asset, tickers, h)
            per_asset_out[asset][str(h)] = dict(
                segments=[s.name for s in res.scenarios],
                medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
                null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
                candidate_dispersion=round(res.candidate_dispersion, 3),
                null_dispersion=round(res.null_dispersion, 3),
                verdict=res.overall_verdict,
                n_folds=res.n_folds,
            )
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        hold_periods=list(HOLD_PERIODS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset_out,
        universe=dict(
            hold_sums={str(h): dict(
                verdict_counts=hold_sums[str(h)]["verdict_counts"],
                ticker_order=hold_sums[str(h)]["ticker_order"],
                per_asset=dict(
                    (t, dict(
                        segments=hold_sums[str(h)]["per_asset"][t]["segments"],
                        medians=hold_sums[str(h)]["per_asset"][t]["medians"],
                        null_medians=hold_sums[str(h)]["per_asset"][t]["null_medians"],
                        candidate_dispersion=hold_sums[str(h)]["per_asset"][t]["candidate_dispersion"],
                        null_dispersion=hold_sums[str(h)]["per_asset"][t]["null_dispersion"],
                        verdict=hold_sums[str(h)]["per_asset"][t]["verdict"],
                    ))
                    for t in ticker_order
                ),
            ) for h in HOLD_PERIODS},
            hold_profiles={t: {str(h): (
                "STABLE_EDGE" if all(
                    profiles[t][str(h)]["verdict"] == "REGIME_STABLE"
                    and all(m > 0 for m in profiles[t][str(h)]["medians"])
                ) else (
                    "SINGULAR" if (
                        sum(
                            profiles[t][str(h2)]["verdict"] == "REGIME_STABLE"
                            and all(m > 0 for m in profiles[t][str(h2)]["medians"])
                        ) == 1 for h2 in HOLD_PERIODS
                    ) else (
                        "WEAKENS" if sum(
                            profiles[t][str(h2)]["verdict"] == "REGIME_STABLE"
                            and all(m > 0 for m in profiles[t][str(h2)]["medians"])
                        ) > 1 else "NO_EDGE"
                    )
                )
            ) for h in HOLD_PERIODS} for t in ticker_order},
            effect_classification=effect_class,
            n_edge_at_h1=n_edge_at_h1,
            n_edge_min_across_grid=n_edge_all,
            stable_edge_assets=[t for t in ticker_order if all(
                profiles[t][str(h)]["verdict"] == "REGIME_STABLE"
                and all(m > 0 for m in profiles[t][str(h)]["medians"])
                for h in HOLD_PERIODS
            )],
            singular_assets=[t for t in ticker_order if any(
                sum(
                    profiles[t][str(h2)]["verdict"] == "REGIME_STABLE"
                    and all(m > 0 for m in profiles[t][str(h2)]["medians"])
                ) == 1 for h2 in HOLD_PERIODS
            )],
            ticker_order=ticker_order,
        ),
        perturbation=pers,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Holding-period sensitivity of lookback-5 momentum: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
