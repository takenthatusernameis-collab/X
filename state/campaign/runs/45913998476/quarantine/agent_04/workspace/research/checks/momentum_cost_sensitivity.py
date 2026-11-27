"""Bounded transaction-cost sensitivity of the short-horizon momentum edge.

Objective (task R-001): decide whether the admitted candidate positive evidence
-- short-horizon momentum (long the previous N-day return, hold 1 day, daily
rebalance) -- survives conservative transaction-cost stress on the collected
real-data universe, judged against a matched coin-flip null.

Why this cell: the momentum class was admitted as candidate positive evidence
(activations 37318950814 / 37361000967) but every published momentum figure was
produced at ZERO transaction cost (``stress_segments`` with the default
``BacktestConfig``). Momentum is a very high-turnover mechanical rule (it
rebalances every bar), so execution cost is the single most decision-relevant
unresolved qualification of the candidate. This check runs a *bounded*,
predeclared cost grid and classifies cost robustness; it does not tune the
signal, the lookback, or the universe after seeing results.

Predeclared cost grid (fixed a-priori, NOT tuned to outcomes):
  zero       : BacktestConfig() -- the reference; reproduces the admitted
               zero-cost momentum evidence exactly.
  realistic  : the repository's existing realistic cost model, copied verbatim
               from examples/ma_crossover_real_data.py:
               commission_per_trade=2.0, commission_per_share=0.003,
               slippage_cents=2.0, slippage_proportional=0.0005.
  stress2x   : a conservative 2x stress -- every cost component doubled
               (commission_per_trade=4.0, commission_per_share=0.006,
               slippage_cents=4.0, slippage_proportional=0.0010).

Matched null: ``stress_segments`` runs the candidate parameter sweep AND the
coin-flip ``noise_benchmark`` under the SAME ``cfg`` (same commission/slippage),
so the null pays the same costs as the candidate. The cost-robustness question
is therefore about the candidate-minus-null gap (the signal's value net of the
mechanical cost drag both legs pay), not about the absolute P&L alone. Absolute
net-of-cost medians are also reported because a tradable candidate must clear
costs in absolute terms too.

Lookbacks: {3, 5, 10} (the predeclared short-horizon set; 5 is canonical / the
admitted lookback). Universe: the fixed 10-asset collected universe. Walk-forward
train=252d/test=84d/warmup=60d/overlap=60d, 4 contiguous volatility blocks,
min 400 bars/segment -- identical to the admitted momentum runs.

A-priori prediction (stated BEFORE execution, falsifiable): because momentum
rebalances every bar, realistic costs are expected to materially erode the net
edge; the candidate-minus-null gap is expected to shrink and the absolute
net-of-cost medians to fall, so the expected classification is COST_MARGINAL or
COST_FRAGILE at the realistic level (the edge is NOT expected to be preserved
unconditionally). COST_ROBUST would refute this prediction.

Cost-robustness classification (predeclared, per lookback), using the per-segment
candidate-minus-null gaps:
  E0 / Er / E2 = set of assets whose gap is strictly positive in EVERY segment
                 at zero / realistic / 2x cost.
  NO_ZERO_COST_EDGE : E0 is empty (no zero-cost edge to stress).
  COST_ROBUST       : E0 non-empty and E0 subset of Er (realistic costs preserve
                      the entire zero-cost edge set).
  COST_MARGINAL     : Er is a non-empty strict subset of E0 (some zero-cost edges
                      survive realistic costs, some do not).
  COST_FRAGILE      : E0 non-empty but Er empty (realistic costs eliminate the
                      edge everywhere).
The overall classification is taken at the canonical lookback 5; the 3 and 10
columns are reported for generality.

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

# Predeclared bounded cost grid (name -> BacktestConfig keyword arguments).
COST_LEVELS = [
    ("zero", {}),
    ("realistic", dict(
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005,
    )),
    ("stress2x", dict(
        commission_per_trade=4.0, commission_per_share=0.006,
        slippage_cents=4.0, slippage_proportional=0.0010,
    )),
]
COST_NAMES = [name for name, _ in COST_LEVELS]

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def make_cfg(cost_kwargs):
    """BacktestConfig for one cost level (warmup set for parity with the repo)."""
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM, **cost_kwargs)


def run_cell(tickers, lookback, cost_kwargs):
    """One (lookback, cost) cell across the universe via the authoritative path.

    Returns per-ticker dict with segments, candidate medians, matched-null
    medians, per-segment candidate-minus-null gaps, dispersions and verdict.
    """
    cfg = make_cfg(cost_kwargs)
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
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    out = {}
    for t in summary.ticker_order:
        r = summary.assets[t]
        cand_raw = [float(s.baseline_median_log_return) for s in r.scenarios]
        null_raw = [float(s.noise_median_log_return) for s in r.scenarios]
        gaps_raw = [c - n for c, n in zip(cand_raw, null_raw)]
        out[t] = dict(
            segments=[s.name for s in r.scenarios],
            medians=[round(c, 6) for c in cand_raw],
            null_medians=[round(n, 6) for n in null_raw],
            gaps=[round(g, 6) for g in gaps_raw],
            candidate_dispersion=round(float(r.candidate_dispersion), 6),
            null_dispersion=round(float(r.null_dispersion), 6),
            verdict=r.overall_verdict,
            n_folds=r.n_folds,
        )
    return out


def edge_set(cell_by_ticker):
    """Assets whose candidate-minus-null gap is strictly positive in every segment."""
    return {t for t, r in cell_by_ticker.items() if all(g > 0 for g in r["gaps"])}


def abs_positive_set(cell_by_ticker):
    """Assets whose absolute net-of-cost candidate medians are all > 0."""
    return {t for t, r in cell_by_ticker.items() if all(m > 0 for m in r["medians"])}


def classify_cell(cell_by_ticker):
    """Predeclared cost-robustness classification for one lookback column.

    Primary (signal value): candidate-minus-null gap > 0 in every segment.
    Secondary (tradability): absolute net-of-cost candidate medians all > 0.
    """
    E0 = edge_set(cell_by_ticker["zero"])
    Er = edge_set(cell_by_ticker["realistic"])
    E2 = edge_set(cell_by_ticker["stress2x"])
    if not E0:
        cls = "NO_ZERO_COST_EDGE"
    elif E0 <= Er:
        cls = "COST_ROBUST"
    elif Er:
        cls = "COST_MARGINAL"
    else:
        cls = "COST_FRAGILE"
    A0 = abs_positive_set(cell_by_ticker["zero"])
    Ar = abs_positive_set(cell_by_ticker["realistic"])
    A2 = abs_positive_set(cell_by_ticker["stress2x"])
    return dict(
        classification=cls,
        E0_zero=sorted(E0),
        Er_realistic=sorted(Er),
        E2_stress2x=sorted(E2),
        n_E0=len(E0), n_Er=len(Er), n_E2=len(E2),
        abs_positive_zero=sorted(A0),
        abs_positive_realistic=sorted(Ar),
        abs_positive_stress2x=sorted(A2),
        n_abs_positive_zero=len(A0),
        n_abs_positive_realistic=len(Ar),
        n_abs_positive_stress2x=len(A2),
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

    print("\n=== 3. Leakage review + cost-accounting audit (AMZN, JPM) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=CANONICAL_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        # Zero-cost reference run.
        res0 = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res0.equity_curve, res0.trades, closes, 1e6)
        # Realistic-cost run: the engine's equity must still be fully explained
        # by the recorded fills (commission + slippage accounting is correct).
        res1 = bt.run_bars(list(bars), signals, make_cfg(dict(COST_LEVELS[1][1])))
        bt.check_equity_matches_fills(res1.equity_curve, res1.trades, closes, 1e6)
        tr0 = (res0.equity_curve[-1] - 1e6) / 1e6
        tr1 = (res1.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, signals {len(signals)}, leakage [PASS], "
              f"equity-fill audit [PASS] (zero & realistic), "
              f"full-sample total_return zero {tr0:+.3f} vs realistic {tr1:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; the "
          "engine fills at close + fixed + proportional slippage with per-trade "
          "and per-share commission, applied to both candidate and null.")

    print("\n=== 4. Cost grid (predeclared) ===")
    for name, kw in COST_LEVELS:
        print(f"  {name:10s}: {kw if kw else 'BacktestConfig() (zero cost)'}")
    print(f"  lookbacks {LOOKBACKS} (canonical {CANONICAL_LOOKBACK}); "
          f"walk-forward train={TRAIN}/test={TEST}/warmup={WARM}/overlap={OVERLAP}; "
          f"{N_BLOCKS} volatility blocks; min {MIN_SEGMENT_BARS} bars/segment\n")

    all_tickers = {}
    for entry in manifest["entries"]:
        all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(all_tickers.keys())

    print("=== 5. Cost-sensitivity grid across collected universe ===")
    # results[lookback][cost][ticker]
    results = {lb: {name: run_cell(all_tickers, lb, kw) for name, kw in COST_LEVELS}
               for lb in LOOKBACKS}

    per_lookback = {}
    for lb in LOOKBACKS:
        cls = classify_cell(results[lb])
        per_lookback[str(lb)] = cls
        print(f"  lookback {lb}: {cls['classification']}  "
              f"|E0|={cls['n_E0']} |Er|={cls['n_Er']} |E2|={cls['n_E2']}  "
              f"E0={cls['E0_zero']}")
    overall = per_lookback[str(CANONICAL_LOOKBACK)]["classification"]
    n_robust_lookbacks = sum(
        1 for lb in LOOKBACKS
        if per_lookback[str(lb)]["classification"] == "COST_ROBUST"
    )
    print(f"\n  OVERALL (canonical lookback {CANONICAL_LOOKBACK}): {overall}")
    print(f"  lookbacks classified COST_ROBUST: {n_robust_lookbacks}/{len(LOOKBACKS)}")

    print("\n  Per-asset detail at lookback 5 (candidate medians / gap vs matched null):")
    for ticker in ticker_order:
        z = results[5]["zero"][ticker]
        r = results[5]["realistic"][ticker]
        s = results[5]["stress2x"][ticker]
        fmt = lambda xs: "[" + ", ".join(f"{x:+.4f}" for x in xs) + "]"
        print(f"    {ticker:6s} zero gap={fmt(z['gaps'])} med={fmt(z['medians'])} [{z['verdict']}]")
        print(f"    {'':6s} real gap={fmt(r['gaps'])} med={fmt(r['medians'])} [{r['verdict']}]")
        print(f"    {'':6s} 2x   gap={fmt(s['gaps'])} med={fmt(s['medians'])} [{s['verdict']}]")
    print("")

    # Determinism: rerun the AMZN/JPM subset over the full grid and compare.
    res2 = {lb: {name: run_cell(tickers, lb, kw) for name, kw in COST_LEVELS}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {name: results[lb][name][t] for name in COST_NAMES for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps({lb: {name: res2[lb][name][t] for name in COST_NAMES for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    det = out1 == out2
    print("determinism: r1 == r2: {}".format(det))
    assert det, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        canonical_lookback=CANONICAL_LOOKBACK,
        cost_levels=[dict(name=name, kwargs=kw) for name, kw in COST_LEVELS],
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        overall_classification=overall,
        per_lookback=per_lookback,
        results={str(lb): results[lb] for lb in LOOKBACKS},
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