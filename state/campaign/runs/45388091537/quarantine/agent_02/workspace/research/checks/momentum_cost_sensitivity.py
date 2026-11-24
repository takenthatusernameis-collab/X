"""Bounded transaction-cost sensitivity check for short-horizon momentum.

Task contract (campaign run 45388091537, agent_02, task R-001):
Does the short-horizon momentum edge survive conservative transaction-cost
stress on the collected real-data universe?

Method (fixed a-priori, not tuned to any outcome):
- Universe: the existing 10-asset collected adjusted-close universe
  (research/data/manifest.json, dataset yf-ohlcv-universe-2009-to-2026-10-03).
- Signal: the existing `bt.momentum_signals` (long the previous N-day return,
  hold 1 day, daily rebalance), lookbacks 3 / 5 / 10 -- unchanged from
  `momentum_lookback_sweep.py`.
- Gate: the existing real-data regime-stability gate (`bt.stress_segments`,
  past-only volatility blocks, walk-forward train=252/test=84/warmup=60/
  overlap=60, min 400 bars/segment) with a matched coin-flip null on the same
  segments, exactly as in the prior momentum checks.
- Cost grid (predeclared, bounded, three levels, applied identically to the
  candidate and the matched null):
    * "zero"          : BacktestConfig()  (reproduces the prior lookback sweep)
    * "realistic"     : the repository's existing realistic cost model
                        (examples/ma_crossover_real_data.py section 7):
                        commission_per_trade=2.0, commission_per_share=0.003,
                        slippage_cents=2.0, slippage_proportional=0.0005
    * "conservative2x": 2x every realistic parameter (bounded upper stress).

A-priori prediction (declared before the grid is evaluated; derived from the
signal definition, not from any cost result): on this universe the momentum
signal is long nearly every day (persistent up-trend), so daily turnover is low
and transaction costs are second order; the realistic-cost per-fold medians
should stay within a small tolerance of the zero-cost medians, no zero-cost
ROBUST asset should flip to a loss / noise verdict at realistic cost, and the
universe-level classification should be COST_ROBUST. The check records whether
this prediction matched.

Cost-robustness classification (predeclared thresholds):
- Per asset and cost level, edge counts across lookbacks {3,5,10} (EDGE = a
  REGIME_STABLE verdict with all-positive segment medians; LOSS =
  REGIME_STABLE_LOSS; NO_EDGE otherwise), and a robustness verdict
  (ROBUST = >=2 EDGE lookbacks, SENSITIVE = exactly 1, NO_EDGE = 0).
- E0 = assets ROBUST at zero cost.
- Universe classification at realistic cost:
    COST_ROBUST   if >= 50% of E0 remain ROBUST,
    COST_FRAGILE  if 0 of E0 remain ROBUST,
    COST_SENSITIVE otherwise.
- The conservative2x column is reported as a stress bound, not used for the
  primary classification.

Turnover diagnostics (full-sample `run_bars` at each cost level) are recorded
so the classification is explained by evidence (turnover, not assertion).

Determinism is asserted via an internal independent recomputation of the
AMZN/JPM subset.

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
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
PRIOR_LOOKBACK_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

# Predeclared bounded cost grid. "realistic" is the repository's existing
# realistic model (examples/ma_crossover_real_data.py, section 7).
COST_GRID = {
    "zero": dict(),
    "realistic": dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    ),
    "conservative2x": dict(
        commission_per_trade=4.0,
        commission_per_share=0.006,
        slippage_cents=4.0,
        slippage_proportional=0.0010,
    ),
}
COST_ORDER = ["zero", "realistic", "conservative2x"]

A_PRIORI_PREDICTION = (
    "COST_ROBUST: on this universe the momentum signal is long nearly every day "
    "(persistent up-trend) so turnover is low; realistic-cost per-fold medians "
    "stay close to zero-cost medians, no zero-cost ROBUST asset flips to a "
    "loss/noise verdict at realistic cost, and >=50% of zero-cost ROBUST assets "
    "remain ROBUST."
)
PREDICTED_CLASSIFICATION = "COST_ROBUST"
COST_ROBUST_FRACTION_THRESHOLD = 0.5


def make_cfg(cost_name):
    return bt.BacktestConfig(**COST_GRID[cost_name])


def regime_result(ticker, bars, lookback, cfg):
    """Regime-stability result for one ticker/lookback under a cost config."""
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
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
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


def signal_diagnostics(bars, lookback, cfg):
    """Full-sample turnover / return diagnostics for the signal at a cost level.

    The signal itself is cost-independent; only the realized fills (and hence
    turnover and total return) depend on the cost config.
    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    sig = bt.momentum_signals(closes, lookback=lookback)
    res = bt.run_bars(list(bars), sig, cfg)
    fills = res.trades
    notional = float(sum(abs(f.shares) * f.price for f in fills))
    avg_eq = float(np.mean(res.equity_curve)) if len(res.equity_curve) else 0.0
    turnover = notional / avg_eq if avg_eq > 0 else 0.0
    total_return = float(res.equity_curve[-1] / res.equity_curve[0] - 1.0) \
        if len(res.equity_curve) and res.equity_curve[0] != 0 else 0.0
    return dict(
        n_trades=len(fills),
        turnover_ratio=round(turnover, 4),
        total_return=round(total_return, 4),
    )


def signal_weights(closes, lookback):
    sig = bt.momentum_signals(closes, lookback=lookback)
    return [s.weight for s in sig]


def edge_of(sr):
    """Per-lookback edge label from a regime result dict (unrounded medians)."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness(edge_by_lb):
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[edge_by_lb[str(lb)]] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts


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
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    ticker_order = [entry["ticker"] for entry in manifest["entries"]]

    print("\n=== 4. Cost grid (predeclared, bounded) ===")
    for name in COST_ORDER:
        print(f"  {name:14s}: {COST_GRID[name]}")
    print(f"\n  A-PRIORI PREDICTION: {A_PRIORI_PREDICTION}")

    # ---- Run the full grid: regime results + turnover diagnostics ----
    print("\n=== 5. Momentum regime-stability across cost grid "
          "(lookbacks 3/5/10 x cost levels) ===")
    # results[ticker][lookback][cost] = regime result
    results = {t: {lb: {} for lb in LOOKBACKS} for t in ticker_order}
    diagnostics = {t: {lb: {} for lb in LOOKBACKS} for t in ticker_order}
    for ticker in ticker_order:
        bars = tickers[ticker]
        closes = bars.closes_array()
        w5 = signal_weights(closes, 5)
        nz = [w for w in w5 if w != 0.0]
        frac_long = float(np.mean([w > 0 for w in nz])) if nz else float("nan")
        flips = int(sum(1 for i in range(1, len(nz)) if nz[i] != nz[i - 1])) if len(nz) > 1 else 0
        buyhold = float(closes[-1] / closes[0] - 1.0)
        for lb in LOOKBACKS:
            for cname in COST_ORDER:
                cfg = make_cfg(cname)
                results[ticker][lb][cname] = regime_result(ticker, bars, lb, cfg)
                diagnostics[ticker][lb][cname] = signal_diagnostics(bars, lb, cfg)
            diagnostics[ticker][lb]["signal"] = dict(
                frac_long=round(frac_long, 3),
                flips=flips,
                nonzero=len(nz),
                buyhold_return=round(buyhold, 4),
            )
        # compact per-asset printout
        for lb in LOOKBACKS:
            med_line = "  ".join(
                "{}:{:+.3f}/{}".format(
                    cname,
                    results[ticker][lb][cname]["medians"][0],
                    results[ticker][lb][cname]["verdict"][:4],
                )
                for cname in COST_ORDER
            )
            print(f"  {ticker:6s} lb={lb:2d} frac_long={frac_long:.3f} "
                  f"flips={flips} | {med_line}")

    # ---- Per-cost-level universe summary (edge counts / robustness) ----
    print("\n=== 6. Per-cost-level robustness summary ===")
    universe = {"per_cost": {}}
    for cname in COST_ORDER:
        per_asset = {}
        rob_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
        verdict5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                           "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
        for ticker in ticker_order:
            edge_by_lb = {str(lb): edge_of(results[ticker][lb][cname]) for lb in LOOKBACKS}
            rob, counts = robustness(edge_by_lb)
            rob_counts[rob] += 1
            verdict5_counts[results[ticker][5][cname]["verdict"]] += 1
            per_asset[ticker] = dict(
                edge_counts=counts,
                robustness_verdict=rob,
                lookback_medians={str(lb): results[ticker][lb][cname]["medians"] for lb in LOOKBACKS},
                lookback_null_medians={str(lb): results[ticker][lb][cname]["null_medians"] for lb in LOOKBACKS},
                verdicts={str(lb): results[ticker][lb][cname]["verdict"] for lb in LOOKBACKS},
                turnover={str(lb): diagnostics[ticker][lb][cname]["turnover_ratio"] for lb in LOOKBACKS},
                n_trades={str(lb): diagnostics[ticker][lb][cname]["n_trades"] for lb in LOOKBACKS},
            )
        universe["per_cost"][cname] = dict(
            robustness_counts=rob_counts,
            lookback_5_verdict_counts=verdict5_counts,
            ticker_order=ticker_order,
            per_asset=per_asset,
        )
        print(f"  {cname:14s}: robustness={rob_counts} | lookback5_verdicts={verdict5_counts}")

    # ---- Cost-robustness classification ----
    print("\n=== 7. Cost-robustness classification ===")
    zero_rob = {t: universe["per_cost"]["zero"]["per_asset"][t]["robustness_verdict"]
                for t in ticker_order}
    real_rob = {t: universe["per_cost"]["realistic"]["per_asset"][t]["robustness_verdict"]
                for t in ticker_order}
    cons_rob = {t: universe["per_cost"]["conservative2x"]["per_asset"][t]["robustness_verdict"]
                for t in ticker_order}
    E0 = [t for t in ticker_order if zero_rob[t] == "ROBUST"]
    real_robust = [t for t in E0 if real_rob[t] == "ROBUST"]
    cons_robust = [t for t in E0 if cons_rob[t] == "ROBUST"]
    n_e0 = len(E0)
    frac_preserved = (len(real_robust) / n_e0) if n_e0 else 0.0
    if n_e0 == 0:
        classification = "COST_FRAGILE"
    elif frac_preserved >= COST_ROBUST_FRACTION_THRESHOLD:
        classification = "COST_ROBUST"
    elif len(real_robust) == 0:
        classification = "COST_FRAGILE"
    else:
        classification = "COST_SENSITIVE"

    # Per-asset change flags for the E0 set (median sign preservation).
    per_asset_change = {}
    for t in E0:
        sign_preserved_real = True
        for lb in LOOKBACKS:
            m0 = results[t][lb]["zero"]["medians"][0]
            mr = results[t][lb]["realistic"]["medians"][0]
            if (m0 > 0) != (mr > 0) and not (m0 == 0 or mr == 0):
                sign_preserved_real = False
        per_asset_change[t] = dict(
            zero_robustness=zero_rob[t],
            realistic_robustness=real_rob[t],
            conservative_robustness=cons_rob[t],
            zero_robust=zero_rob[t] == "ROBUST",
            realistic_robust=real_rob[t] == "ROBUST",
            conservative_robust=cons_rob[t] == "ROBUST",
            median_sign_preserved_real=bool(sign_preserved_real),
        )

    prediction_matched = (classification == PREDICTED_CLASSIFICATION)

    print(f"  E0 (ROBUST at zero cost): {E0}")
    print(f"  ROBUST at realistic: {real_robust} ({len(real_robust)}/{n_e0})")
    print(f"  ROBUST at conservative2x: {cons_robust} ({len(cons_robust)}/{n_e0})")
    print(f"  fraction preserved (realistic): {frac_preserved:.3f} "
          f"(threshold {COST_ROBUST_FRACTION_THRESHOLD})")
    print(f"  CLASSIFICATION: {classification}")
    print(f"  a-priori prediction matched: {prediction_matched} "
          f"(predicted {PREDICTED_CLASSIFICATION})")

    # ---- Cross-consistency: zero-cost lookback=5 vs prior verified artifact ----
    print("\n=== 8. Cross-consistency vs prior lookback sweep (zero cost, lookback 5) ===")
    xcheck_ok = True
    if PRIOR_LOOKBACK_ARTIFACT.exists():
        with open(PRIOR_LOOKBACK_ARTIFACT) as f:
            prior = json.load(f)
        for ticker in ticker_order:
            pub = prior["per_asset"][ticker]["5"]["medians"]
            mine = results[ticker][5]["zero"]["medians"]
            status = "MATCH" if pub == mine else "MISMATCH"
            if status != "MATCH":
                xcheck_ok = False
                print(f"  [MISMATCH] {ticker}: mine {mine} vs prior {pub}")
        print(f"  zero-cost lookback-5 medians vs prior verified artifact: "
              f"{'ALL MATCH' if xcheck_ok else 'MISMATCH FOUND'}")
    else:
        print("  prior lookback artifact not found; cross-consistency SKIPPED")
        xcheck_ok = None

    # ---- Determinism: rerun AMZN/JPM subset and assert identical ----
    res2 = {t: {lb: {c: regime_result(t, tickers[t], lb, make_cfg(c)) for c in COST_ORDER}
                for lb in LOOKBACKS}
            for t in ["AMZN", "JPM"]}
    out1 = json.dumps({t: {lb: results[t][lb] for lb in LOOKBACKS} for t in ["AMZN", "JPM"]},
                      sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    deterministic = out1 == out2
    print(f"\n=== 9. Determinism: r1 == r2: {deterministic} ===")
    assert deterministic, "non-deterministic output"

    # ---- Write artifact ----
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_grid={k: COST_GRID[k] for k in COST_ORDER},
        cost_order=COST_ORDER,
        a_priori_prediction=A_PRIORI_PREDICTION,
        predicted_classification=PREDICTED_CLASSIFICATION,
        cost_robust_fraction_threshold=COST_ROBUST_FRACTION_THRESHOLD,
        per_asset={
            t: {
                str(lb): {
                    cname: results[t][lb][cname] for cname in COST_ORDER
                }
                for lb in LOOKBACKS
            }
            for t in ticker_order
        },
        signal_diagnostics={
            t: {str(lb): diagnostics[t][lb] for lb in LOOKBACKS}
            for t in ticker_order
        },
        universe=universe,
        cost_robustness=dict(
            e0_zero_robust=E0,
            realistic_robust=real_robust,
            conservative_robust=cons_robust,
            n_e0=n_e0,
            fraction_preserved_realistic=round(frac_preserved, 4),
            classification=classification,
            prediction_matched=prediction_matched,
            per_asset_change=per_asset_change,
        ),
        cross_consistency_prior_zero_lookback5=xcheck_ok,
        determinism=deterministic,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum transaction-cost sensitivity on the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())