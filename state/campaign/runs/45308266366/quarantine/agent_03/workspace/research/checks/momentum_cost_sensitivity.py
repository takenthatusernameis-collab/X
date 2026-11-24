"""Bounded transaction-cost sensitivity check for short-horizon momentum.

Primary question (R-001): does the short-horizon momentum edge survive
conservative transaction-cost stress on the collected real-data universe?

Design (fixed a-priori, not tuned to results):

1. Signal-integrity gate. The momentum artifacts report *identical* candidate
   medians across lookbacks 3/5/10 for every asset, which a genuine
   "long the previous N-day return" signal cannot do. This check therefore
   first audits what ``research.backtest.momentum_signals`` actually computes.
   The contracted implementation evaluates ``sign(mean(log(close)))`` over the
   lookback window -- the sign of an average *price level*, not a return. Since
   every collected adjusted close is far above 1.0, that expression is always
   positive, so the contracted signal is (near) always-long and is effectively
   lookback-independent. It is therefore not a momentum signal, and a cost test
   on it is not a valid test of momentum cost robustness (its turnover is ~0).

2. Cost grid (predeclared): zero, realistic (the repository's existing
   realistic cost model), and conservative (2x the realistic variable costs).
   The realistic model is the one used by the repository's examples:
   commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0,
   slippage_proportional=0.0005.

3. For each signal variant, lookback in {3, 5, 10}, and cost level, the
   momentum portfolio is run through the existing walk-forward regime-stability
   gate (``stress_segments``) with a matched coin-flip null (the same cost model
   is applied to the null, so candidate and null are directly comparable).
   Per-asset verdicts are aggregated into a count of assets with a uniformly
   positive REGIME_STABLE edge.

4. Two signal variants are evaluated on the identical grid:
   - "contracted": ``research.backtest.momentum_signals`` as implemented (the
     literal target of the task); documented as degenerate.
   - "corrected": a genuine momentum signal defined a-priori from the class
     docstring -- +1 if the previous ``lookback``-day log return is positive,
     else -1. This is the signal the prior research *intended* to test, and is
     the basis for the cost-robustness classification.

5. Predeclared cost-robustness classification (primary signal = corrected,
   lookback = 5):
   - N0 = # assets with a positive REGIME_STABLE edge at zero cost.
   - If N0 == 0 -> NO_ZERO_COST_EDGE (no edge to stress).
   - Else majority = N0 // 2 + 1:
       N_realistic >= majority            -> COST_ROBUST
       1 <= N_realistic <  majority       -> COST_DEGRADED
       N_realistic == 0                   -> COST_COLLAPSES

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = (3, 5, 10)
PRIMARY_LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

# Predeclared cost grid. "realistic" is the repository's existing realistic cost
# model (examples/ma_crossover.py, examples/ma_crossover_real_data.py,
# examples/volatility_regime_filter.py). "conservative" doubles the variable
# costs as a bounded stress above realistic.
COST_LEVELS = {
    "zero": {},
    "realistic": {
        "commission_per_trade": 2.0,
        "commission_per_share": 0.003,
        "slippage_cents": 2.0,
        "slippage_proportional": 0.0005,
    },
    "conservative": {
        "commission_per_trade": 4.0,
        "commission_per_share": 0.006,
        "slippage_cents": 4.0,
        "slippage_proportional": 0.0010,
    },
}


def make_cfg(cost):
    return bt.BacktestConfig(initial_capital=1e6, **cost)


def corrected_momentum_signals(closes, lookback=5):
    """Genuine momentum: +1 if the previous `lookback`-day log return > 0 else
    -1 (0 before the window). This is the textbook definition stated in the
    momentum_signals docstring, provided here as an independent code path."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lg = np.log(closes)
    lb = int(lookback)
    for i in range(lb, n):
        r = lg[i] - lg[i - lb]
        out[i] = bt.Signal(date=i + 1, weight=1.0 if r > 0 else -1.0)
    return out


def avg_turnover(signals):
    """Mean absolute daily change in position weight (per-bar trading intensity
    that a transaction-cost model charges on)."""
    w = np.array([s.weight for s in signals], dtype=np.float64)
    if len(w) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(w))))


def signal_integrity(tickers):
    """Audit the contracted momentum_signals vs a genuine momentum signal."""
    per_ticker = {}
    degenerate = True
    for ticker in _ticker_order(tickers):
        bars = tickers[ticker]
        closes = bars.closes_array()
        entry = {}
        for lb in LOOKBACKS:
            sig = bt.momentum_signals(closes, lookback=lb)
            w = np.array([s.weight for s in sig], dtype=np.float64)[lb:]
            distinct = sorted(set(np.round(w, 6).tolist()))
            frac_long = float(np.mean(w > 0))
            corr = corrected_momentum_signals(closes, lb)
            cw = np.array([s.weight for s in corr], dtype=np.float64)[lb:]
            active = (w != 0) & (cw != 0)
            sign_match = (float(np.mean(np.sign(w[active]) == np.sign(cw[active])))
                          if active.any() else None)
            entry[str(lb)] = dict(
                distinct_weights=distinct,
                frac_long=round(frac_long, 4),
                sign_match_corrected=round(sign_match, 4) if sign_match is not None else None,
                turnover_contract=round(avg_turnover(sig), 6),
                turnover_corrected=round(avg_turnover(corr), 6),
            )
            # A genuine momentum signal must take both signs and be lookback
            # sensitive; flag degeneracy per (ticker, lookback).
            if not (len(distinct) == 2 and 0.2 < frac_long < 0.8):
                degenerate = True
        per_ticker[ticker] = entry
    # Degenerate if the contracted signal is (near) always-long for the
    # majority of tickers at the primary lookback.
    always_long = sum(
        1 for t in per_ticker
        if per_ticker[t][str(PRIMARY_LOOKBACK)]["frac_long"] >= 0.99
    )
    return dict(
        definition="sign(mean(log(close[i-lookback+1:i+1]))) -- sign of an "
                   "average price level, not a return",
        per_ticker=per_ticker,
        always_long_tickers_at_primary=always_long,
        n_tickers=len(per_ticker),
        degenerate=bool(always_long >= (len(per_ticker) + 1) // 2),
    )


def _ticker_order(tickers):
    return list(tickers.keys())


def run_cell(tickers, signals_fn, lookback, cost):
    """Run the walk-forward regime-stability gate across the universe for one
    (variant, lookback, cost) cell, with a matched coin-flip null."""
    summary = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=signals_fn,
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", lookback),),
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=make_cfg(cost),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    per_asset = {}
    positive_edge = 0
    for ticker in summary.ticker_order:
        r = summary.assets[ticker]
        medians = [round(s.baseline_median_log_return, 3) for s in r.scenarios]
        nulls = [round(s.noise_median_log_return, 3) for s in r.scenarios]
        is_edge = (r.overall_verdict == "REGIME_STABLE"
                   and all(m > 0 for m in medians))
        if is_edge:
            positive_edge += 1
        per_asset[ticker] = dict(
            segments=[s.name for s in r.scenarios],
            medians=medians,
            null_medians=nulls,
            candidate_dispersion=round(r.candidate_dispersion, 3),
            null_dispersion=round(r.null_dispersion, 3),
            n_folds=r.n_folds,
            verdict=r.overall_verdict,
            positive_edge=bool(is_edge),
        )
    return dict(
        per_asset=per_asset,
        verdict_counts=summary.verdict_counts,
        positive_edge=positive_edge,
    )


def evaluate_variant(tickers, signals_fn):
    """Evaluate one signal variant over the full (lookback x cost) grid."""
    out = {}
    for lb in LOOKBACKS:
        out[str(lb)] = {}
        for cost_name, cost in COST_LEVELS.items():
            print("    variant={} lookback={} cost={}".format(
                signals_fn.__name__, lb, cost_name))
            out[str(lb)][cost_name] = run_cell(tickers, signals_fn, lb, cost)
    return out


def classify(n0, nr):
    """Predeclared cost-robustness classification (corrected signal, lb=5)."""
    if n0 == 0:
        return "NO_ZERO_COST_EDGE"
    majority = n0 // 2 + 1
    if nr >= majority:
        return "COST_ROBUST"
    if nr >= 1:
        return "COST_DEGRADED"
    return "COST_COLLAPSES"


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print("  {}: {}".format(
            entry["ticker"], "OK" if actual == entry["checksum_sha256"] else "MISMATCH"))
    print("  dataset: {} | universe: {}".format(
        manifest["dataset_id"], len(manifest["universe"])))

    print("\n=== 2. Data preflight ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    # Load the full universe once.
    tickers = {}
    for entry in manifest["entries"]:
        t = entry["ticker"]
        tickers[t] = bt.load_ticker(t)[0]
    ticker_order = list(tickers.keys())

    print("\n=== 3. Signal-integrity gate (contracted momentum_signals) ===")
    integrity = signal_integrity(tickers)
    print("  contracted signal definition: {}".format(integrity["definition"]))
    print("  always-long tickers at lookback {}: {}/{}".format(
        PRIMARY_LOOKBACK, integrity["always_long_tickers_at_primary"],
        integrity["n_tickers"]))
    for t in ticker_order:
        e = integrity["per_ticker"][t][str(PRIMARY_LOOKBACK)]
        print("    {:6s} distinct={} frac_long={:.3f} sign_match={} "
              "turn_contract={:.4f} turn_corrected={:.4f}".format(
                  t, e["distinct_weights"], e["frac_long"],
                  e["sign_match_corrected"], e["turnover_contract"],
                  e["turnover_corrected"]))
    print("  DEGENERATE (not a momentum signal): {}".format(integrity["degenerate"]))

    print("\n=== 4. Leakage review (corrected momentum, AMZN/JPM) ===")
    for t in ["AMZN", "JPM"]:
        bars = tickers[t]
        closes = bars.closes_array()
        sig = corrected_momentum_signals(closes, PRIMARY_LOOKBACK)
        bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), sig,
                          bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print("  {}: {} bars, corrected momentum signals {}, leakage [PASS]".format(
            t, bars.n_bars, len(sig)))
    print("  no-look-ahead: signal at bar t uses log(close[t]) - log(close[t-lookback]) only.")

    print("\n=== 5. Cost grid x lookback x variant (matched coin-flip null) ===")
    print("  cost levels: {}".format({k: v for k, v in COST_LEVELS.items()}))
    contracted = evaluate_variant(tickers, bt.momentum_signals)
    corrected = evaluate_variant(tickers, corrected_momentum_signals)

    # Cost-robustness classification (corrected signal, primary lookback 5).
    n0 = corrected[str(PRIMARY_LOOKBACK)]["zero"]["positive_edge"]
    nr = corrected[str(PRIMARY_LOOKBACK)]["realistic"]["positive_edge"]
    nc = corrected[str(PRIMARY_LOOKBACK)]["conservative"]["positive_edge"]
    classification = classify(n0, nr)

    # Also record the contracted signal's cost sensitivity (for the record).
    c0 = contracted[str(PRIMARY_LOOKBACK)]["zero"]["positive_edge"]
    cr = contracted[str(PRIMARY_LOOKBACK)]["realistic"]["positive_edge"]
    cc = contracted[str(PRIMARY_LOOKBACK)]["conservative"]["positive_edge"]

    print("\n  Corrected momentum, lookback {} positive-edge assets: "
          "zero={} realistic={} conservative={} -> {}".format(
              PRIMARY_LOOKBACK, n0, nr, nc, classification))
    print("  Contracted (degenerate) signal, lookback {} positive-edge assets: "
          "zero={} realistic={} conservative={}".format(PRIMARY_LOOKBACK, c0, cr, cc))

    print("\n=== 6. Determinism (AMZN/JPM, lookback 5, zero + realistic) ===")
    det_ok = True
    for variant_name, fn in (("contracted", bt.momentum_signals),
                             ("corrected", corrected_momentum_signals)):
        for cost_name in ("zero", "realistic"):
            r1 = run_cell({t: tickers[t] for t in ["AMZN", "JPM"]}, fn,
                          PRIMARY_LOOKBACK, COST_LEVELS[cost_name])
            r2 = run_cell({t: tickers[t] for t in ["AMZN", "JPM"]}, fn,
                          PRIMARY_LOOKBACK, COST_LEVELS[cost_name])
            same = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
            det_ok = det_ok and same
            print("  {} {}: r1==r2 -> {}".format(variant_name, cost_name, same))
    assert det_ok, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        primary_lookback=PRIMARY_LOOKBACK,
        cost_levels=COST_LEVELS,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        signal_integrity=integrity,
        cells=dict(contracted=contracted, corrected=corrected),
        cost_robustness=dict(
            primary_signal="corrected",
            lookback=PRIMARY_LOOKBACK,
            n0_zero_cost_positive_edge=n0,
            n_realistic_positive_edge=nr,
            n_conservative_positive_edge=nc,
            classification=classification,
            rule="N0==0 -> NO_ZERO_COST_EDGE; else majority=N0//2+1: "
                 "N_realistic>=majority -> COST_ROBUST; "
                 "1<=N_realistic<majority -> COST_DEGRADED; "
                 "N_realistic==0 -> COST_COLLAPSES",
            contracted_signal=dict(
                degenerate=integrity["degenerate"],
                zero=c0, realistic=cr, conservative=cc,
                note="contracted signal is (near) always-long with ~0 turnover; "
                     "cost impact is negligible, so this is NOT a valid test of "
                     "momentum cost robustness",
            ),
        ),
        determinism=bool(det_ok),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())