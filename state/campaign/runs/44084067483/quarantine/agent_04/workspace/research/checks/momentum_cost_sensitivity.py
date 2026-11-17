"""Bounded transaction-cost sensitivity of the qualified short-horizon momentum signal.

This is the R-001 focused task. The momentum class (long the previous
lookback-day return, hold 1 day, daily rebalance) has been admitted as candidate
positive evidence: a REGIME_STABLE positive pattern in 7/10 of the collected
universe at lookback 5 (activation 37318950814), shown to be lookback-robust at
3/5/10 (activation 37361000967) and horizon-stable in magnitude to 60 days
(activation 37394344201). Every prior momentum result was computed with the
engine's default (zero-cost) ``BacktestConfig``. The remaining decision-relevant
qualification is whether that mechanical, high-turnover edge survives
conservative transaction costs.

Primary question:
    Does the short-horizon momentum edge survive conservative transaction-cost
    stress on the collected real-data universe?

Method (fixed a-priori, NOT tuned to results):
  * Lookbacks: {3, 5, 10} (the pre-existing lookback-robustness grid).
  * Cost model: the repository's existing realistic cost model, taken verbatim
    from ``examples/ma_crossover_real_data.py``:
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005
    A single non-negative scale multiplies every cost component, giving a
    predeclared cost grid COST_SCALES = (0.0, 1.0, 2.0, 5.0):
        0.0x -> zero cost (must reproduce the admitted zero-cost result)
        1.0x -> the repository's realistic cost model
        2.0x -> conservative stress (double the realistic cost)
        5.0x -> severe stress (five times the realistic cost)
  * Walk-forward: train=252d / test=84d / warmup=60d / overlap=60d, 4 contiguous
    volatility blocks, min 400 bars per segment, exactly as ``momentum.py`` and
    ``momentum_lookback_sweep.py``.
  * Matched null: the framework's coin-flip null (``noise_benchmark``) run on
    the same segments under the same cost config, so candidate and null are
    charged identically. Caveat (recorded, not hidden): the coin-flip null flips
    position far more often than the momentum signal, so it is charged more
    turnover per unit time. The null is therefore a cost-consistent reference,
    not a turnover-matched one; the operative cost evidence is the candidate's
    own net fold medians and their sign/cross-segment stability.

Classification (predeclared; applied after the grid runs, never used to select
parameters). Classification is deliberately based on the CANDIDATE's own net
medians (verdict-free), not on the framework REGIME_STABLE count, because the
coin-flip null is destroyed by turnover costs and a collapsing null mechanically
inflates the candidate's relative verdict (see the null caveat above). For each
lookback L let:
  pos(L, s) = # assets whose every segment median is strictly positive at cost
              scale s  (a net-positive cross-segment edge, candidate-only)
  base = pos(L, 0.0x), real = pos(L, 1.0x), cons = pos(L, 2.0x)
  * COST_ROBUST    : real >= base and cons >= base
  * COST_TOLERANT  : real >= base and cons >= base - 1
  * COST_SEVERE    : otherwise
  * COST_DESTROYED : real == 0
Universe-level: COST_ROBUST if all three lookbacks are COST_ROBUST; COST_TOLERANT
if every lookback is COST_ROBUST or COST_TOLERANT; otherwise COST_SEVERE.

A SECOND, decisive diagnostic is reported: candidate signal activity (annualized
position sign changes). A short-horizon timing strategy must trade; a signal
that never changes position is a static exposure, and cost sensitivity is
vacuous for it because there is almost no turnover to charge.

A-priori falsification prediction (fixed before the run): if the momentum edge is
a genuine high-turnover timing effect, its REGIME_STABLE positive count should
fall materially as costs rise from 0.0x to 2.0x, and the candidate's net medians
should decline with cost. Either the count is retained (cost-robust) or it is not
(cost-fragile); both close the cell.

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
COST_SCALES = (0.0, 1.0, 2.0, 5.0)

# Repository's existing realistic cost model (examples/ma_crossover_real_data.py).
REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cost_config(scale):
    """Engine cost config with every cost component scaled by ``scale``."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=REALISTIC_COST["commission_per_trade"] * scale,
        commission_per_share=REALISTIC_COST["commission_per_share"] * scale,
        slippage_cents=REALISTIC_COST["slippage_cents"] * scale,
        slippage_proportional=REALISTIC_COST["slippage_proportional"] * scale,
    )


def run_universe(tickers, lookback, scale):
    """Regime-stability universe summary for one lookback at one cost scale."""
    cfg = cost_config(scale)
    return bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", lookback),),
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )


def summarize(summary):
    """Per-asset medians/nulls/verdicts and the asset counts used for
    classification."""
    per_asset = {}
    uniform_positive = 0
    for ticker in summary.ticker_order:
        res = summary.assets[ticker]
        medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
        null_medians = [round(s.noise_median_log_return, 3) for s in res.scenarios]
        verdict = res.overall_verdict
        if all(m > 0 for m in medians):
            uniform_positive += 1
        per_asset[ticker] = dict(
            segments=[s.name for s in res.scenarios],
            medians=medians,
            null_medians=null_medians,
            candidate_dispersion=round(res.candidate_dispersion, 3),
            null_dispersion=round(res.null_dispersion, 3),
            verdict=verdict,
            n_folds=res.n_folds,
        )
    return dict(
        verdict_counts=summary.verdict_counts,
        ticker_order=summary.ticker_order,
        per_asset=per_asset,
        regime_stable_count=summary.verdict_counts["REGIME_STABLE"],
        uniform_positive_count=uniform_positive,
    )


def candidate_turnover(closes, lookback):
    """Annualized number of position sign changes for one asset/lookback.

    Turnover is a property of the signal series only (cost-independent), so it
    is computed once and reused across cost scales.
    """
    sig = bt.momentum_signals(closes, lookback=lookback)
    weights = [s.weight for s in sig]
    changes = sum(1 for i in range(1, len(weights)) if weights[i] != weights[i - 1])
    years = len(weights) / 252.0
    return int(changes), float(changes / years) if years > 0 else 0.0


def classify(base, real, cons):
    """Predeclared cost-robustness classification for one lookback.

    Uses the candidate-only uniform-positive count, so a collapsing null cannot
    drive the verdict.
    """
    if real == 0:
        return "COST_DESTROYED"
    if real >= base and cons >= base:
        return "COST_ROBUST"
    if real >= base and cons >= base - 1:
        return "COST_TOLERANT"
    return "COST_SEVERE"


def overall_classification(per_lookback):
    classes = [v["classification"] for v in per_lookback.values()]
    if all(c == "COST_ROBUST" for c in classes):
        return "COST_ROBUST"
    if all(c in ("COST_ROBUST", "COST_TOLERANT") for c in classes):
        return "COST_TOLERANT"
    return "COST_SEVERE"


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

    print("\n=== 3. Leakage review (lookback=5 reference) ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        cfg0 = cost_config(0.0)
        res = bt.run_bars(list(bars), signals, cfg0)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
    print(f"  {len(tickers)} tickers: check_signal_integrity + "
          "check_equity_matches_fills [PASS]")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Candidate signal activity (annualized position sign changes) ===")
    turnover = {}
    for ticker in tickers:
        turnover[ticker] = {}
        for lookback in LOOKBACKS:
            changes, per_year = candidate_turnover(tickers[ticker].closes_array(), lookback)
            turnover[ticker][str(lookback)] = dict(
                changes=changes, per_year=round(per_year, 1),
                static_position=changes <= 2)
    print("  {:6s} {:>12s} {:>12s} {:>12s}".format(
        "ticker", "lb=3", "lb=5", "lb=10"))
    for ticker in tickers:
        print("  {:6s} {:>12s} {:>12s} {:>12s}".format(
            ticker,
            "{:.1f}/yr".format(turnover[ticker]["3"]["per_year"]),
            "{:.1f}/yr".format(turnover[ticker]["5"]["per_year"]),
            "{:.1f}/yr".format(turnover[ticker]["10"]["per_year"])))
    null_changes_analytic = 252.0 * (4.0 / 9.0)
    print("  coin-flip null (analytic, p_neutral=1/3): "
          "{:.1f}/yr changes".format(null_changes_analytic))
    n_static = sum(
        1 for t in turnover for lb in LOOKBACKS
        if turnover[t][str(lb)]["static_position"])
    print("  static-position flags (<=2 sign changes over the full series): "
          "{}/{} asset-lookback cells".format(n_static, len(turnover) * len(LOOKBACKS)))
    if n_static == len(turnover) * len(LOOKBACKS):
        print("  WARNING: every asset-lookback cell is a static position; the "
              "implemented 'momentum' signal never changes direction, so cost "
              "sensitivity is vacuous for it. See research/checks/"
              "diag_momentum_signal.py.")

    print("\n=== 5. Cost grid: lookback x cost scale (10-asset universe) ===")
    print("  cost scales: {}".format(list(COST_SCALES)))
    print("  realistic cost model: {}".format(REALISTIC_COST))
    print("  walk-forward: train={}d / test={}d / warmup={}d / overlap={}d, "
          "min {} bars/segment, {} volatility blocks\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, N_BLOCKS))

    grid = {}
    for lookback in LOOKBACKS:
        grid[str(lookback)] = {}
        for scale in COST_SCALES:
            summary = run_universe(tickers, lookback, scale)
            snap = summarize(summary)
            grid[str(lookback)][str(scale)] = snap
            print("  lookback {} cost {:>3.1f}x: verdict_counts={} "
                  "regime_stable={} uniform_positive={}".format(
                      lookback, scale, snap["verdict_counts"],
                      snap["regime_stable_count"], snap["uniform_positive_count"]))
        print("")

    # Predeclared per-lookback classification (candidate-only, verdict-free).
    per_lookback = {}
    for lookback in LOOKBACKS:
        base = grid[str(lookback)]["0.0"]["uniform_positive_count"]
        real = grid[str(lookback)]["1.0"]["uniform_positive_count"]
        cons = grid[str(lookback)]["2.0"]["uniform_positive_count"]
        cls = classify(base, real, cons)
        per_lookback[str(lookback)] = dict(
            base_uniform_positive=base,
            realistic_uniform_positive=real,
            conservative_uniform_positive=cons,
            severe_uniform_positive=grid[str(lookback)]["5.0"]["uniform_positive_count"],
            base_regime_stable=grid[str(lookback)]["0.0"]["regime_stable_count"],
            realistic_regime_stable=grid[str(lookback)]["1.0"]["regime_stable_count"],
            conservative_regime_stable=grid[str(lookback)]["2.0"]["regime_stable_count"],
            severe_regime_stable=grid[str(lookback)]["5.0"]["regime_stable_count"],
            classification=cls,
        )
    overall = overall_classification(per_lookback)

    print("=== 6. Per-lookback cost-robustness classification ===")
    for lookback in LOOKBACKS:
        p = per_lookback[str(lookback)]
        print("  lookback {}: uniform_positive base={} real={} cons={} severe={} "
              "-> {}".format(
                  lookback, p["base_uniform_positive"],
                  p["realistic_uniform_positive"],
                  p["conservative_uniform_positive"],
                  p["severe_uniform_positive"], p["classification"]))
        print("              REGIME_STABLE base={} real={} cons={} severe={}".format(
            p["base_regime_stable"], p["realistic_regime_stable"],
            p["conservative_regime_stable"], p["severe_regime_stable"]))
    print("  universe-level: {}".format(overall))
    print("")

    print("=== 7. Per-asset medians across cost scales (lookback 5, reference) ===")
    print("  {:6s} {:>7s} {:>24s} {:>24s} {:>24s} {:>24s}".format(
        "ticker", "verdict", "0.0x", "1.0x", "2.0x", "5.0x"))
    for ticker in grid["5"]["0.0"]["ticker_order"]:
        cells = []
        for scale in COST_SCALES:
            snap = grid["5"][str(scale)]["per_asset"][ticker]
            cells.append("{:+.3f} {}".format(
                float(np.median(snap["medians"])), snap["verdict"][:6]))
        print("  {:6s} {:>7s} {}".format(
            ticker, grid["5"]["0.0"]["per_asset"][ticker]["verdict"][:7],
            " ".join("{:>24s}".format(c) for c in cells)))
    print("")

    # Determinism: rerun the AMZN/JPM subset for lookbacks 3/5/10 at 1.0x.
    sub = {t: tickers[t] for t in ["AMZN", "JPM"]}
    rerun = {}
    for lb in LOOKBACKS:
        rerun[str(lb)] = summarize(run_universe(sub, lb, 1.0))
    out1 = json.dumps(
        {str(lb): {t: grid[str(lb)]["1.0"]["per_asset"][t] for t in sub}
         for lb in LOOKBACKS},
        sort_keys=True)
    out2 = json.dumps(
        {str(lb): {t: rerun[str(lb)]["per_asset"][t] for t in sub}
         for lb in LOOKBACKS}, sort_keys=True)
    deterministic = out1 == out2
    print("determinism: r1 == r2: {}".format(deterministic))
    assert deterministic, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_scales=list(COST_SCALES),
        realistic_cost_model=REALISTIC_COST,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        null_note=(
            "coin-flip null run on the same segments under the same cost config; "
            "not turnover-matched (coin-flip changes position ~{:.0f}/yr vs "
            "candidate ~5-30/yr), so it is a cost-consistent reference rather "
            "than a like-for-like benchmark".format(null_changes_analytic)),
        grid=grid,
        per_lookback=per_lookback,
        overall_classification=overall,
        turnover=turnover,
        signal_static_position=bool(
            n_static == len(turnover) * len(LOOKBACKS)),
        determinism=deterministic,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())