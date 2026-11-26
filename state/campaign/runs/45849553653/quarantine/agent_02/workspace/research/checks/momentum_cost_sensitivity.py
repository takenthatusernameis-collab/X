"""Bounded transaction-cost sensitivity of short-horizon momentum (R-001).

Background: short-horizon momentum (long the previous N-day return, hold 1 day,
daily rebalance) is admitted as candidate positive evidence on the collected
universe (`research/checks/momentum.py`, activation 37318950814; lookback
robustness in `research/checks/momentum_lookback_sweep.py`, activation
37361000967). The admission is explicitly qualified: the edge is short-horizon
and mechanical, and execution-cost robustness was left as a decision-relevant
qualification. This check closes that qualification.

Primary question: does the lookback 3/5/10 momentum edge survive conservative
transaction-cost stress on the collected real-data universe?

Design (fixed a-priori, not tuned to results):
- Universe: the existing 10 collected tickers (unchanged).
- Signal: the existing `bt.momentum_signals` (unchanged), lookbacks {3, 5, 10}.
- Gate: the existing `bt.stress_segments` walk-forward regime-stability gate
  (train=252d / test=84d / warmup=60d / overlap=60d, 4 contiguous volatility
  blocks, min 400 bars/segment), unchanged from the momentum admission.
- Matched null: the framework's own coin-flip null (`noise_benchmark`) run on
  the same segments with the SAME cost config, so the null pays the same
  transaction costs as the candidate (a cost-matched null).
- Cost grid (predeclared, bounded, three levels):
    zero      : BacktestConfig() (no commission, no slippage) -- reproduces the
                prior zero-cost momentum result and serves as the reference.
    realistic : commission_per_trade=2.0, commission_per_share=0.003,
                slippage_cents=2.0, slippage_proportional=0.0005 -- the
                repository's existing realistic cost model
                (`examples/ma_crossover_real_data.py`).
    double    : exactly 2x the realistic model -- a conservative stress level.

A-priori prediction (declared BEFORE execution, falsifiable): because the rule
rebalances daily, realistic costs apply on nearly every bar; I predict the
absolute net-of-cost edge is materially compressed and that most cells that are
a positive REGIME_STABLE edge at zero cost fall to CONSISTENT_WITH_NOISE or
REGIME_STABLE_LOSS at the realistic level (i.e. the edge is COST_FRAGILE for
most assets). Because the matched null pays the same costs, the candidate-minus-
null gap should be largely preserved even where the absolute edge is not; that
gap is reported so the signal's information content is not conflated with the
common cost drag. The prediction is FALSIFIED if the majority of zero-cost EDGE
cells remain EDGE at the realistic level.

Cost-robustness classification (per asset x lookback cell), using the same edge
definition as the momentum admission (REGIME_STABLE with uniformly positive
medians = EDGE; REGIME_STABLE_LOSS = LOSS; otherwise NO_EDGE):
- COST_ROBUST : EDGE at zero AND EDGE at realistic.
- COST_FRAGILE: EDGE at zero but NOT EDGE at realistic.
- NO_EDGE     : not EDGE at zero.
A per-asset overall class is COST_ROBUST if >= 2 of its 3 lookback cells are
COST_ROBUST, COST_FRAGILE if any cell is COST_FRAGILE (and < 2 COST_ROBUST),
else NO_EDGE.

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

# Predeclared cost grid (do not tune after seeing results).
COST_LEVELS = {
    "zero": bt.BacktestConfig(),
    "realistic": bt.BacktestConfig(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    ),
    "double": bt.BacktestConfig(
        commission_per_trade=4.0,
        commission_per_share=0.006,
        slippage_cents=4.0,
        slippage_proportional=0.0010,
    ),
}


def edge_status(medians, verdict):
    """EDGE / LOSS / NO_EDGE for one cell, matching the momentum admission."""
    if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
        return "EDGE"
    if verdict == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def cost_class(edge_zero, edge_realistic):
    if edge_zero == "EDGE" and edge_realistic == "EDGE":
        return "COST_ROBUST"
    if edge_zero == "EDGE" and edge_realistic != "EDGE":
        return "COST_FRAGILE"
    return "NO_EDGE"


def signal_profile(closes, lookback):
    """Characterize the existing momentum signal's weights (data provenance).

    Reports the non-neutral weight composition so the cost result can be
    interpreted: a genuine short-horizon momentum signal should be roughly
    balanced between +1 and -1, whereas a degenerate (e.g. always-long)
    implementation will be almost entirely one sign. This characterizes the
    signal; it does not modify it.
    """
    sig = bt.momentum_signals(closes, lookback=lookback)
    n = len(sig)
    n_pos = sum(1 for s in sig if s.weight > 0)
    n_neg = sum(1 for s in sig if s.weight < 0)
    n_neu = n - n_pos - n_neg
    nonneutral = n_pos + n_neg
    return dict(
        n_bars=n,
        n_long=n_pos,
        n_short=n_neg,
        n_neutral=n_neu,
        frac_long=round(n_pos / nonneutral, 4) if nonneutral else 0.0,
        frac_short=round(n_neg / nonneutral, 4) if nonneutral else 0.0,
    )


def run_cell(ticker, bars, lookback, cfg):
    """One (asset, lookback, cost) stress_segments run -> plain dict."""
    closes = bars.closes_array()
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
    medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
    nulls = [round(s.noise_median_log_return, 3) for s in res.scenarios]
    gaps = [round(m - n, 3) for m, n in zip(medians, nulls)]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=medians,
        null_medians=nulls,
        gaps=gaps,
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
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

    print("\n=== 3. Leakage review per asset (lookback=5) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
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

    print("\n=== 3b. Signal characterization (weight composition, lookback=5) ===")
    signal_profiles = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars_t = bt.load_ticker(ticker)[0]
        prof = signal_profile(bars_t.closes_array(), 5)
        signal_profiles[ticker] = prof
        print("  {:6s} long={:5d} short={:5d} neutral={:4d} "
              "frac_long={:.3f} frac_short={:.3f}".format(
                  ticker, prof["n_long"], prof["n_short"], prof["n_neutral"],
                  prof["frac_long"], prof["frac_short"]))
    n_degenerate = sum(1 for p in signal_profiles.values() if p["frac_long"] > 0.99)
    print("  assets effectively always-long (frac_long>0.99): {}/10".format(n_degenerate))
    print("  NOTE: a genuine short-horizon momentum signal should be roughly")
    print("  balanced between long and short; a near-100% long composition means")
    print("  the tested signal is (near-)constant long exposure, not a direction bet.")

    print("\n=== 4. Cost grid (predeclared) ===")
    for name, cfg in COST_LEVELS.items():
        print(f"  {name:10s}: commission_per_trade={cfg.commission_per_trade} "
              f"commission_per_share={cfg.commission_per_share} "
              f"slippage_cents={cfg.slippage_cents} "
              f"slippage_proportional={cfg.slippage_proportional}")
    print("  matched null: framework coin-flip null run with the SAME cost config "
          "on the same segments (cost-matched).")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    print("\n=== 5. Momentum cost-sensitivity sweep (lookback 3/5/10 x cost grid) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 contiguous volatility blocks\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    per_cell = {}
    for lookback in LOOKBACKS:
        for ticker in ticker_order:
            bars = all_tickers[ticker]
            levels = {name: run_cell(ticker, bars, lookback, cfg)
                      for name, cfg in COST_LEVELS.items()}
            ez = edge_status(levels["zero"]["medians"], levels["zero"]["verdict"])
            er = edge_status(levels["realistic"]["medians"], levels["realistic"]["verdict"])
            ed = edge_status(levels["double"]["medians"], levels["double"]["verdict"])
            cell = dict(levels)
            cell["edge_zero"] = ez
            cell["edge_realistic"] = er
            cell["edge_double"] = ed
            cell["cost_class"] = cost_class(ez, er)
            cell["median_gap_zero"] = round(float(np.median(levels["zero"]["gaps"])), 3)
            cell["median_gap_realistic"] = round(float(np.median(levels["realistic"]["gaps"])), 3)
            cell["median_gap_double"] = round(float(np.median(levels["double"]["gaps"])), 3)
            per_cell[f"{lookback}|{ticker}"] = cell

    # Per-asset aggregation across lookbacks.
    per_ticker = {}
    class_counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "NO_EDGE": 0}
    cell_class_counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "NO_EDGE": 0}
    cells_edge_zero = cells_edge_realistic = cells_edge_double = 0
    for ticker in ticker_order:
        classes = []
        for lookback in LOOKBACKS:
            c = per_cell[f"{lookback}|{ticker}"]
            classes.append(c["cost_class"])
            cell_class_counts[c["cost_class"]] += 1
            cells_edge_zero += 1 if c["edge_zero"] == "EDGE" else 0
            cells_edge_realistic += 1 if c["edge_realistic"] == "EDGE" else 0
            cells_edge_double += 1 if c["edge_double"] == "EDGE" else 0
        n_robust = classes.count("COST_ROBUST")
        if n_robust >= 2:
            overall = "COST_ROBUST"
        elif "COST_FRAGILE" in classes:
            overall = "COST_FRAGILE"
        else:
            overall = "NO_EDGE"
        class_counts[overall] += 1
        per_ticker[ticker] = dict(
            lookback_classes={str(lb): per_cell[f"{lb}|{ticker}"]["cost_class"]
                              for lb in LOOKBACKS},
            overall=overall,
            median_gap_realistic={str(lb): per_cell[f"{lb}|{ticker}"]["median_gap_realistic"]
                                  for lb in LOOKBACKS},
        )

    for ticker in ticker_order:
        p = per_ticker[ticker]
        detail = " ".join(
            "{}:{}[z={:+.3f} r={:+.3f} d={:+.3f}]".format(
                lb,
                p["lookback_classes"][str(lb)][:4],
                per_cell[f"{lb}|{ticker}"]["zero"]["medians"][0],
                per_cell[f"{lb}|{ticker}"]["realistic"]["medians"][0],
                per_cell[f"{lb}|{ticker}"]["double"]["medians"][0],
            )
            for lb in LOOKBACKS
        )
        print(f"  {ticker:6s} overall={p['overall']:12s} {detail}")
    print("")
    print(f"  cells EDGE at zero={cells_edge_zero}/30  realistic={cells_edge_realistic}/30  "
          f"double={cells_edge_double}/30")
    print(f"  cell cost_class counts: {cell_class_counts}")
    print(f"  per-asset overall counts: {class_counts}")

    # Determinism: rerun the AMZN/JPM subset at lookback 5, realistic cost.
    res2 = {t: run_cell(t, tickers[t], 5, COST_LEVELS["realistic"]) for t in target}
    out1 = json.dumps({t: per_cell[f"5|{t}"]["realistic"] for t in target}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    det = out1 == out2
    print("\ndeterminism: r1 == r2 (AMZN/JPM lookback5 realistic): {}".format(det))
    assert det, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_levels={
            name: dict(
                commission_per_trade=cfg.commission_per_trade,
                commission_per_share=cfg.commission_per_share,
                slippage_cents=cfg.slippage_cents,
                slippage_proportional=cfg.slippage_proportional,
            )
            for name, cfg in COST_LEVELS.items()
        },
        a_priori_prediction=(
            "Most zero-cost EDGE cells fall to non-EDGE at the realistic cost level "
            "(COST_FRAGILE); the candidate-minus-null gap is largely preserved "
            "because the matched null pays the same costs. FALSIFIED if the "
            "majority of zero-cost EDGE cells remain EDGE at realistic."
        ),
        signal_profile_note="frac_long near 1.0 => degenerate near-constant-long signal, not a direction bet; cost class then describes a constant-long position.",
        signal_profile=signal_profiles,
        per_cell=per_cell,
        summary=dict(
            cells_edge_zero=cells_edge_zero,
            cells_edge_realistic=cells_edge_realistic,
            cells_edge_double=cells_edge_double,
            cell_class_counts=cell_class_counts,
            per_asset_overall=class_counts,
            per_ticker=per_ticker,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())