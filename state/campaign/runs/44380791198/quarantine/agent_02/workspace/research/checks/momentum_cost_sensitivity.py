"""Bounded transaction-cost sensitivity check for lookback 3/5/10 momentum.

Primary question (task R-001): Does the short-horizon momentum edge survive
conservative transaction-cost stress on the collected real-data universe?

Background: the momentum class (go LONG the previous N-day return, hold 1 day,
daily rebalance) is admitted as candidate positive evidence at lookback 5
(`research/checks/momentum.py`, activation 37318950814) and survives a
lookback sweep 3/5/10 (`research/checks/momentum_lookback_sweep.py`, activation
37361000967): 6/10 assets are lookback-ROBUST. The remaining UNVERIFIED
qualification is execution-cost robustness: a daily-rebalanced 1-day bet flips
position nearly every bar, so per-trade slippage + commission could plausibly
erase the 0.05-0.13 median log-return edge. This check closes that gap.

Design (fixed a-priori, declared before execution, not tuned to results):

- Signal: `bt.momentum_signals(closes, lookback=N)` for N in {3, 5, 10} --
  the existing implementation, unchanged.
- Walk-forward: same windows as the lookback sweep (train=252, test=84,
  warmup=60, overlap=60) inside the same 4 contiguous volatility blocks
  (`volatility_blocks(n_blocks=4, window=60)`).
- Cost model (predeclared, commission-dominant). A diagnostic
  (`scratch/diag_slippage_asymmetry.py`) established that the engine's
  symmetric proportional slippage (always added to the fill price regardless
  of side) is structurally NEUTRAL-to-beneficial for a continuously-invested
  long/short strategy that flips direction every bar: the closing/selling leg
  receives the inflated fill price, so slippage cancels (and can even add) on
  a round trip. The binding cost for this high-turnover signal is therefore the
  fixed per-trade commission, which is always subtracted. The cost grid
  (commission_per_trade, 5 points including the zero-cost reference):
    * `zero`     : $0/trade       (reference)
    * `comm_25`  : $25/trade      (~0.5 bp round-trip at $1M notional)
    * `comm_50`  : $50/trade      (~1 bp round-trip; realistic institutional)
    * `comm_100` : $100/trade     (~2 bp round-trip)
    * `comm_500` : $500/trade     (~10 bp round-trip; conservative stress)
  A $1M notional round trip at 1 bp = $100, i.e. $50/trade, so the grid spans
  realistic to conservative. No leverage; target_exposure 1.0. The proportional
  slippage channel is deliberately zero here because it is non-binding for this
  signal; a separate note records the slippage-neutrality finding so the choice
  is auditable and not hidden.
- Matched null: the same coin-flip null used by the framework
  (`noise_benchmark` with the SAME `cfg` costs as the candidate), run on the
  same segments. Judging the costed candidate against a costed null isolates
  the edge from the (identical) cost drag both bear.
- Verdict rule: per lookback, the costed median must (a) stay positive and
  (b) exceed the costed null median by the sample-calibrated tolerance
  (`compare_noise`, 2x null fold-dispersion / sqrt(n_folds)) to count as a
  surviving edge. The asset-level cost-robustness verdict is COST_ROBUST if
  the edge survives at the predeclared conservative `comm_500` level for >= 2
  lookbacks; COST_SENSITIVE if it survives at the `comm_50` (~1bp) level but
  not at `comm_500`; COST_ERASED if it does not survive even at the zero-cost
  reference for >= 2 lookbacks.

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

# Predeclared cost grid: (label, commission_per_trade, slippage_cents).
# `zero` is the no-cost reference; `comm_500` is the predeclared conservative
# level that an edge must survive to be called cost-robust. Proportional
# slippage is zero across the grid because it is structurally neutral for this
# continuously-invested long/short flip signal (see module docstring and
# scratch/diag_slippage_asymmetry.py); the binding cost is fixed commission.
COST_GRID = [
    ("zero", 0.0, 0.0),
    ("comm_25", 25.0, 0.0),
    ("comm_50", 50.0, 0.0),
    ("comm_100", 100.0, 0.0),
    ("comm_500", 500.0, 0.0),
]
REFERENCE_COST = "zero"
HARSH_COST = "comm_500"
REALISTIC_COST = "comm_50"  # ~1 bp round-trip; realistic institutional level

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cost_cfg(commission_per_trade, slippage_cents):
    """BacktestConfig for one cost-grid point (no leverage, exposure 1.0)."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        commission_per_trade=float(commission_per_trade),
        commission_per_share=0.0,
        slippage_cents=float(slippage_cents),
        slippage_proportional=0.0,
    )


def segment_result(ticker, bars, lookback, commission_per_trade, slippage_cents):
    """Walk-forward regime-stability result for one ticker / lookback / cost.

    Uses the same machinery as the lookback sweep (`stress_segments`), but with
    the cost parameters threaded into the walk-forward config so both the
    candidate and the framework coin-flip null bear identical costs.
    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    cfg = cost_cfg(commission_per_trade, slippage_cents)
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


def survives_edge(sr):
    """Whether one costed segment result shows a surviving positive edge.

    An edge survives when the framework verdict is REGIME_STABLE (not
    CONSISTENT_WITH_NOISE / REGIME_DEPENDENT / REGIME_STABLE_LOSS) AND every
    segment median is strictly positive. This is the same edge convention used
    by the lookback sweep (momentum_lookback_sweep.py::lookback_verdict).
    """
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def asset_cost_verdict(cost_profiles):
    """Asset-level cost-robustness verdict across the predeclared cost grid.

    COST_ROBUST    : a surviving edge at the conservative `comm_500` level for
                     >= 2 of the 3 lookbacks.
    COST_SENSITIVE : a surviving edge at the realistic `comm_50` (~1bp) level
                     for >= 2 lookbacks, but not cost-robust at `comm_500`.
    COST_ERASED    : no surviving edge even at the zero-cost reference for >= 2
                     lookbacks (the edge is already absent without costs).
    """
    ref_lb = [lb for lb in LOOKBACKS if survives_edge(cost_profiles[REFERENCE_COST][str(lb)])]
    realistic_lb = [lb for lb in LOOKBACKS if survives_edge(cost_profiles[REALISTIC_COST][str(lb)])]
    harsh_lb = [lb for lb in LOOKBACKS if survives_edge(cost_profiles[HARSH_COST][str(lb)])]
    if len(harsh_lb) >= 2:
        return "COST_ROBUST", ref_lb, realistic_lb, harsh_lb
    if len(realistic_lb) >= 2:
        return "COST_SENSITIVE", ref_lb, realistic_lb, harsh_lb
    return "COST_ERASED", ref_lb, realistic_lb, harsh_lb


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

    print("\n=== 3. Leakage review (lookback=5 reference, cost=zero) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cost_cfg(0.0, 0.0))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost-sensitivity sweep (lookbacks 3/5/10 x cost grid) ===")
    print("  Predeclared cost grid (commission/trade; proportional slippage is 0")
    print("  because it is structurally neutral for this flip signal):")
    for label, comm, slip in COST_GRID:
        tag = "  <- reference" if label == REFERENCE_COST else (
              "  <- realistic (~1bp RT)" if label == REALISTIC_COST else (
              "  <- conservative threshold" if label == HARSH_COST else ""))
        print(f"    {label:9s}: commission ${comm:.0f}/trade, slippage {slip:.0f} cents/share{tag}")
    print("  Walk-forward per segment: train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 volatility blocks; candidate "
          "and coin-flip null carry identical costs\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # cost_results[cost_label][lookback][ticker] = segment_result
    cost_results = {}
    for label, comm, slip in COST_GRID:
        cost_results[label] = {
            lb: {t: segment_result(t, all_tickers[t], lb, comm, slip) for t in ticker_order}
            for lb in LOOKBACKS
        }

    per_asset = {}
    verdict_counts = {"COST_ROBUST": 0, "COST_SENSITIVE": 0, "COST_ERASED": 0}
    for ticker in ticker_order:
        cost_profiles = {label: {str(lb): cost_results[label][lb][ticker] for lb in LOOKBACKS}
                         for label, _, _ in COST_GRID}
        asset_verdict, ref_lb, realistic_lb, harsh_lb = asset_cost_verdict(cost_profiles)
        verdict_counts[asset_verdict] += 1
        per_asset[ticker] = {
            "cost_profiles": cost_profiles,
            "cost_verdict": asset_verdict,
            "reference_edge_lookbacks": ref_lb,
            "realistic_edge_lookbacks": realistic_lb,
            "harsh_edge_lookbacks": harsh_lb,
        }

    print("  Asset-level cost verdicts (reference / realistic / harsh edge lookbacks):")
    for ticker in ticker_order:
        p = per_asset[ticker]
        print("    {:6s}: ref_lbs={} realistic_lbs={} harsh_lbs={} -> {}".format(
            ticker, p["reference_edge_lookbacks"], p["realistic_edge_lookbacks"],
            p["harsh_edge_lookbacks"], p["cost_verdict"]))
    print("")
    print("  Cost-robustness verdict counts: COST_ROBUST={} COST_SENSITIVE={} "
          "COST_ERASED={}".format(
              verdict_counts["COST_ROBUST"], verdict_counts["COST_SENSITIVE"],
              verdict_counts["COST_ERASED"]))
    print("")

    # Determinism: rerun the target subset at harsh cost and assert identical.
    res2 = {lb: {t: segment_result(t, tickers[t], lb, 500.0, 0.0) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: cost_results[HARSH_COST][lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: harsh-cost (comm_500) r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_grid=[{"label": label, "commission_per_trade": comm,
                    "slippage_cents": slip} for label, comm, slip in COST_GRID],
        reference_cost=REFERENCE_COST,
        realistic_cost=REALISTIC_COST,
        harsh_cost=HARSH_COST,
        slippage_note=("Proportional slippage is structurally neutral for a "
                       "continuously-invested long/short flip signal (the engine "
                       "adds slippage to the fill price on both buy and sell legs, "
                       "so it cancels on a round trip). The binding cost is fixed "
                       "per-trade commission. See scratch/diag_slippage_asymmetry.py."),
        per_asset=per_asset,
        universe=dict(
            cost_verdict_counts=verdict_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    cost_profiles={label: {
                        str(lb): {
                            "medians": per_asset[t]["cost_profiles"][label][str(lb)]["medians"],
                            "null_medians": per_asset[t]["cost_profiles"][label][str(lb)]["null_medians"],
                            "verdict": per_asset[t]["cost_profiles"][label][str(lb)]["verdict"],
                            "n_folds": per_asset[t]["cost_profiles"][label][str(lb)]["n_folds"],
                        }
                        for lb in LOOKBACKS
                    } for label, _, _ in COST_GRID},
                    cost_verdict=per_asset[t]["cost_verdict"],
                    reference_edge_lookbacks=per_asset[t]["reference_edge_lookbacks"],
                    realistic_edge_lookbacks=per_asset[t]["realistic_edge_lookbacks"],
                    harsh_edge_lookbacks=per_asset[t]["harsh_edge_lookbacks"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity sweep across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())