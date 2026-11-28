"""Real-data transaction-cost sensitivity of the short-horizon momentum edge.

Objective (focused task R-001, campaign slot 02): run a bounded cost-sensitivity
check for lookback 3/5/10 momentum under the repository's existing realistic
cost model and matched null, and classify cost robustness.

Background: activation 37318950814 admitted short-horizon momentum (long the
previous N-day return, hold 1 day, daily rebalance) as candidate positive
evidence (REGIME_STABLE positive edge in 7/10 assets at lookback 5); activation
37361000967 showed the edge is lookback-robust across 3/5/10 for 6/10 assets.
The remaining decision-relevant qualification is execution cost: a daily
rebalanced 1-day-hold bet turns over heavily, so a positive gross-of-cost edge
can be an artifact of ignoring turnover. This check adds the cost dimension
without touching the signal rule, the universe, or leverage.

Cost model (the repository's existing realistic model, from
examples/ma_crossover.py and examples/ma_crossover_real_data.py):
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005
The engine applies commission + slippage to every fill (every round trip), and
`stress_segments` forwards the same `cfg` to BOTH the candidate walk-forward and
the coin-flip `noise_benchmark`, so the null is cost-matched on the same
segments.

Predeclared cost grid (a-priori, NOT tuned to results):
    zero        : BacktestConfig() defaults (no cost) -- reference; reproduces
                  the admitted lookback sweep exactly (cross-consistency gate).
    realistic   : the canonical realistic model above.
    conservative: 2x the realistic variable costs (commission_per_trade=4.0,
                  commission_per_share=0.006, slippage_cents=4.0,
                  slippage_proportional=0.001 = 10 bps) -- a conservative stress.

Predeclared classification (per asset), using EDGE = REGIME_STABLE with
uniformly positive segment medians (identical rule to
`momentum_lookback_sweep.lookback_verdict`):
    survives[cost] = number of lookbacks in {3,5,10} that are EDGE at `cost`.
    COST_ROBUST  : survives[conservative] >= 1 (a positive edge persists even
                   under conservative costs).
    COST_FRAGILE : survives[zero] >= 1 but survives[conservative] == 0 (edge is
                   gross-of-cost only).
    NO_EDGE      : survives[zero] == 0 (no positive edge even with zero cost).

The primary question -- does the short-horizon momentum edge survive conservative
transaction-cost stress -- is answered by the universe COST_ROBUST count and by
the lookback-5-specific `lb5_survives_conservative` flag (the admitted cell).

Method (fixed a-priori): for each collected ticker and each lookback in {3,5,10}
and each cost level, run `stress_segments` (walk-forward train=252/test=84/
warmup=60/overlap=60, 4 contiguous past-only volatility blocks, min 400
bars/segment) with the cost `cfg`; record per-segment candidate medians, matched
null medians, dispersions, verdict, and fold count. Determinism is asserted via
an internal independent recomputation of the AMZN subset. Research / simulation
only. No live trading or production execution.
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
COST_LEVELS = ("zero", "realistic", "conservative")
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cost_configs():
    """Ordered, predeclared cost grid (see module docstring)."""
    return {
        "zero": bt.BacktestConfig(initial_capital=1e6),
        "realistic": bt.BacktestConfig(
            initial_capital=1e6,
            commission_per_trade=2.0,
            commission_per_share=0.003,
            slippage_cents=2.0,
            slippage_proportional=0.0005,
        ),
        "conservative": bt.BacktestConfig(
            initial_capital=1e6,
            commission_per_trade=4.0,
            commission_per_share=0.006,
            slippage_cents=4.0,
            slippage_proportional=0.001,
        ),
    }


def cell_result(ticker, bars, lookback, cfg):
    """One (ticker, lookback, cost) stress_segments cell."""
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
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def is_edge(cell):
    """EDGE = REGIME_STABLE with uniformly positive segment medians.

    Identical rule to momentum_lookback_sweep.lookback_verdict. This is the
    framework's significance-aware edge definition (it also requires the
    candidate's cross-segment swing to stay within 2x the null dispersion).
    """
    return cell["verdict"] == "REGIME_STABLE" and all(m > 0 for m in cell["medians"])


def positive_candidate(cell):
    """Candidate-only positivity: every segment median > 0, independent of the
    null. Reported alongside the verdict-based EDGE so a verdict flip driven by
    a null-dispersion change (candidate medians unchanged) is not mistaken for a
    genuine change in the candidate's P&L."""
    return all(m > 0 for m in cell["medians"])


def classify(per_lookback_cost):
    """Cost-robustness classification for one asset.

    per_lookback_cost: {lookback(int): {cost(str): cell}}

    Two complementary views are returned:
      verdict-based (primary, framework EDGE definition):
        survives[cost] = # lookbacks that are EDGE at `cost`.
        COST_ROBUST / COST_FRAGILE / NO_EDGE as documented in the module header.
      candidate-based (diagnostic, null-independent):
        survives_pos[cost] = # lookbacks whose candidate medians are all > 0.
    plus the largest absolute candidate-median shift from zero to conservative
    cost (how much cost actually moves the candidate's own P&L).
    """
    survives = {c: sum(1 for lb in LOOKBACKS if is_edge(per_lookback_cost[lb][c]))
                for c in COST_LEVELS}
    survives_pos = {c: sum(1 for lb in LOOKBACKS if positive_candidate(per_lookback_cost[lb][c]))
                    for c in COST_LEVELS}
    if survives["conservative"] >= 1:
        verdict = "COST_ROBUST"
    elif survives["zero"] >= 1:
        verdict = "COST_FRAGILE"
    else:
        verdict = "NO_EDGE"
    max_shift = 0.0
    for lb in LOOKBACKS:
        z = per_lookback_cost[lb]["zero"]["medians"]
        c = per_lookback_cost[lb]["conservative"]["medians"]
        for a, b in zip(z, c):
            max_shift = max(max_shift, abs(b - a))
    lb5_survives_conservative = is_edge(per_lookback_cost[5]["conservative"])
    return dict(
        verdict=verdict,
        survives=survives,
        survives_pos=survives_pos,
        max_candidate_median_shift=round(max_shift, 4),
        lb5_survives_conservative=lb5_survives_conservative,
    )


def turnover_diagnostic(tickers, lookback, cfg):
    """Full-sample trade count per asset at one lookback/cost (mechanism probe).

    A daily-rebalanced 1-day-hold bet flips position whenever the lookback
    return sign flips; each flip is a full round trip. This records how often
    that happens, to explain why cost bites.
    """
    out = {}
    for ticker, bars in tickers.items():
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=lookback)
        res = bt.run_bars(list(bars), sig, cfg)
        out[ticker] = dict(
            n_bars=int(bars.n_bars),
            n_fills=len(res.trades),
            turnover_per_bar=round(len(res.trades) / max(1, bars.n_bars), 4),
        )
    return out


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cfgs = cost_configs()

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

    print("\n=== 3. Leakage review per asset (lookback=5, realistic cost) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cfgs["realistic"])
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], net total_return(realistic) {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "costs applied per fill by the engine.")

    print("\n=== 4. Cost-sensitivity grid (lookback 3/5/10 x cost zero/realistic/"
          "conservative) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 contiguous volatility blocks, "
          "each cell vs a cost-matched coin-flip null\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # per_asset[ticker][lookback][cost] = cell
    per_asset = {}
    class_counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "NO_EDGE": 0}
    lb5_conservative_edge = 0
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        per_lb_cost = {}
        for lb in LOOKBACKS:
            per_lb_cost[lb] = {}
            for cost in COST_LEVELS:
                per_lb_cost[lb][cost] = cell_result(ticker, bars, lb, cfgs[cost])
        cls = classify(per_lb_cost)
        class_counts[cls["verdict"]] += 1
        if cls["lb5_survives_conservative"]:
            lb5_conservative_edge += 1
        per_asset[ticker] = dict(
            cells={str(lb): per_lb_cost[lb] for lb in LOOKBACKS},
            survives=cls["survives"],
            survives_pos=cls["survives_pos"],
            max_candidate_median_shift=cls["max_candidate_median_shift"],
            cost_robustness=cls["verdict"],
            lb5_survives_conservative=cls["lb5_survives_conservative"],
        )
        # Compact per-asset printout.
        med_line = []
        for lb in LOOKBACKS:
            z = per_lb_cost[lb]["zero"]["medians"][0]
            r = per_lb_cost[lb]["realistic"]["medians"][0]
            c = per_lb_cost[lb]["conservative"]["medians"][0]
            med_line.append("lb{}: z{:+.3f}/r{:+.3f}/c{:+.3f}".format(lb, z, r, c))
        print("  {:6s} {} | survives={} pos={} maxshift={:.4f} -> {}".format(
            ticker, "  ".join(med_line), cls["survives"], cls["survives_pos"],
            cls["max_candidate_median_shift"], cls["verdict"]))
    print("")
    print("  Universe cost-robustness counts (verdict-based): COST_ROBUST={} "
          "COST_FRAGILE={} NO_EDGE={}".format(
              class_counts["COST_ROBUST"], class_counts["COST_FRAGILE"],
              class_counts["NO_EDGE"]))
    print("  lookback-5 (admitted cell) survives conservative cost: {}/10 "
          "assets".format(lb5_conservative_edge))
    # Candidate-only (null-independent) view, to expose verdict flips driven by
    # the null rather than the candidate.
    pos_cons = sum(1 for t in ticker_order if per_asset[t]["survives_pos"]["conservative"] >= 1)
    pos_zero = sum(1 for t in ticker_order if per_asset[t]["survives_pos"]["zero"] >= 1)
    eroded = [t for t in ticker_order
              if per_asset[t]["survives_pos"]["zero"] >= 1
              and per_asset[t]["survives_pos"]["conservative"] == 0]
    print("  Candidate-only (null-independent) positive at zero: {}/10; at "
          "conservative: {}/10; eroded by cost: {}".format(
              pos_zero, pos_cons, eroded if eroded else "none"))
    print("")

    # Turnover mechanism probe (lookback 5, realistic cost).
    turnover = turnover_diagnostic(all_tickers, 5, cfgs["realistic"])
    print("=== 5. Turnover diagnostic (lookback 5, realistic cost) ===")
    for ticker in ticker_order:
        t = turnover[ticker]
        print("  {:6s}: fills={} / {} bars (turnover/bar {:.4f})".format(
            ticker, t["n_fills"], t["n_bars"], t["turnover_per_bar"]))
    print("")

    # Determinism: rerun AMZN lookback 5 across all cost levels and compare.
    det_ref = {cost: cell_result("AMZN", tickers["AMZN"], 5, cfgs[cost])
               for cost in COST_LEVELS}
    det_rerun = {cost: cell_result("AMZN", tickers["AMZN"], 5, cfgs[cost])
                 for cost in COST_LEVELS}
    out1 = json.dumps(det_ref, sort_keys=True)
    out2 = json.dumps(det_rerun, sort_keys=True)
    print("determinism: AMZN lookback5 all-cost r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_levels=list(COST_LEVELS),
        cost_configs={
            "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                         slippage_cents=0.0, slippage_proportional=0.0),
            "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                              slippage_cents=2.0, slippage_proportional=0.0005),
            "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                                 slippage_cents=4.0, slippage_proportional=0.001),
        },
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            class_counts=class_counts,
            lb5_survives_conservative=lb5_conservative_edge,
            candidate_positive_zero=pos_zero,
            candidate_positive_conservative=pos_cons,
            candidate_eroded_by_cost=eroded,
            ticker_order=ticker_order,
        ),
        turnover_diagnostic=turnover,
        determinism=dict(AMZN_lookback5_all_cost_identical=(out1 == out2)),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())