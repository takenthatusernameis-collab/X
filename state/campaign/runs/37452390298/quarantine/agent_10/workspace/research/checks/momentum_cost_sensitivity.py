"""Cost-sensitivity test for short-horizon momentum on the collected universe.

Question: does the short-horizon momentum edge (long the previous N-day
return, hold 1 day, daily rebalance; admitted as candidate positive evidence
at lookback 3/5/10 in research/checks/momentum.py and
research/checks/momentum_lookback_sweep.py) survive conservative
transaction-cost stress on the collected real-data universe?

This is a bounded cost-sensitivity check. It uses the existing momentum
implementation and the existing walk-forward regime-stability gate with a
matched coin-flip null, and it varies only the backtest cost policy.

COST MODEL (fixed a-priori, documented, not tuned to OOS)
---------------------------------------------------------
Costs are implemented through research.backtest.engine.BacktestConfig and
apply to every round trip (fill price = close + fixed_slippage +
proportional_slippage; commission = commission_per_trade +
commission_per_share * abs(position_change); target exposure = 1.0 of the
1e6 initial capital). Four levels:

    zero     slippage 0 bps, commission $0.0000/share  (baseline, matches
             the existing momentum lookback sweep exactly)
    realistic slippage 10 bps per leg, $0.0050/share
             -- standard discount-broker large-cap assumptions; ~20 bps
                round trip plus per-share commission
    conservative slippage 20 bps, $0.0100/share
             -- conservative buffer against slippage-model error and wider
                spread on less-liquid days
    heavy slippage 50 bps, $0.0200/share
             -- order-of-magnitude stress on turnover cost

All levels use initial_capital=1e6, target_exposure=1.0,
commission_per_trade=0, slippage_cents=0. Reference round-trip cost
at a $100 price, per dollar of notional, is printed for interpretation:
    zero    0.000000   realistic  0.002100   conservative  0.004100
    heavy   0.010100

The coin-flip null runs through the same cost policy, so the comparison is
candidate median vs null median at each cost level. Because the momentum
candidate turns over (weight +-1 every bar) more than the random
{ -1, 0, +1 } null (one third neutral), costs do not cancel and the
candidate-vs-null gap is the quantity under test.

Method (fixed a-priori, not tuned to OOS): lookbacks 3 / 5 / 10, the same
walk-forward windows and volatility blocks as
research/checks/momentum_lookback_sweep.py (train=252 / test=84 /
warmup=60 / overlap=60, 4 contiguous volatility blocks, min 400 bars
segment), full collected 10-asset universe, matched null per segment.

Verdicts
--------
Per lookback: EDGE = REGIME_STABLE with uniformly positive medians; LOSS =
REGIME_STABLE_LOSS; otherwise NO_EDGE. Robustness = EDGE at >= 2 lookbacks.

Cost-robustness (new): per asset, at which cost levels a robust EDGE
(EDGE at >= 2 lookbacks) survives:
    cost-ROBUST  : robust EDGE survives at conservative costs
    cost-SENSITIVE: robust EDGE survives at realistic but not conservative
    cost-NO_EDGE : no robust EDGE at conservative costs (edge erased by
                   conservative transaction costs)

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
REFERENCE_PRICE = 100.0

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

# ----------------------------------------------------------------------------
# Cost grid (fixed a-priori, see module docstring)
# ----------------------------------------------------------------------------
COST_LEVELS = {
    "zero": bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0),
    "realistic": bt.BacktestConfig(
        initial_capital=1e6, target_exposure=1.0,
        slippage_proportional=0.001, commission_per_share=0.005),
    "conservative": bt.BacktestConfig(
        initial_capital=1e6, target_exposure=1.0,
        slippage_proportional=0.002, commission_per_share=0.010),
    "heavy": bt.BacktestConfig(
        initial_capital=1e6, target_exposure=1.0,
        slippage_proportional=0.005, commission_per_share=0.020),
}


def ref_cost_per_notional(cfg):
    """Reference round-trip cost as a fraction of notional at REFERENCE_PRICE."""
    shares = 1.0 * cfg.initial_capital * cfg.target_exposure / REFERENCE_PRICE
    return (2.0 * shares * cfg.slippage_proportional * REFERENCE_PRICE
            + 2.0 * shares * cfg.commission_per_share) / (
                cfg.initial_capital * cfg.target_exposure)


# ----------------------------------------------------------------------------
# Core result computation: mirrors the lookback-sweep path, parameterized by
# a cost policy so that only costs change.
# ----------------------------------------------------------------------------
def segment_result(ticker, bars, lookback, cfg):
    """Regime-stability result for one ticker and one lookback under cfg."""
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
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    scenarios = res.scenarios
    medians = [round(s.baseline_median_log_return, 3) for s in scenarios]
    null_medians = [round(s.noise_median_log_return, 3) for s in scenarios]
    gaps = [round(m - n, 3) for m, n in zip(medians, null_medians)]
    return dict(
        segments=[s.name for s in scenarios],
        medians=medians,
        null_medians=null_medians,
        gaps=gaps,
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        n_periods=res.n_periods,
        verdict=res.overall_verdict,
    )


def lookback_verdict(sr):
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST"
    if counts["EDGE"] == 1:
        return "SENSITIVE"
    return "NO_EDGE"


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

    print("\n=== 2. Cost grid (reference round-trip cost per unit notional at $100) ===")
    for name, cfg in COST_LEVELS.items():
        print(f"  {name:11s}: slippage {cfg.slippage_proportional*1e4:6.1f} bps, "
              f"comm ${cfg.commission_per_share*1000:.2f}/share, "
              f"ref cost {ref_cost_per_notional(cfg):.6f}")

    print("\n=== 3. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 4. Leakage review per asset (lookback=5, realistic costs) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, COST_LEVELS["realistic"])
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 5. Momentum cost-sensitivity sweep (3 / 5 / 10) across "
          "collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments, four cost levels\n".format(
          TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # [cost_level][lookback][ticker][segment index].
    results = {
        name: {lb: {t: segment_result(t, all_tickers[t], lb, cfg)
                    for t in ticker_order}
               for lb in LOOKBACKS}
        for name, cfg in COST_LEVELS.items()
    }

    # Cost-robustness classification per asset.
    cost_robustness = {}
    print("  per-asset cost-robustness (edge survives at which cost levels):")
    for ticker in ticker_order:
        survives = []
        for name in ["zero", "realistic", "conservative", "heavy"]:
            edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
            for lb in LOOKBACKS:
                edge_counts[lookback_verdict(results[name][lb][ticker])] += 1
            if edge_counts["EDGE"] >= 2:
                survives.append(name)
        if "conservative" in survives:
            verdict = "cost-ROBUST"
        elif "realistic" in survives:
            verdict = "cost-SENSITIVE"
        else:
            verdict = "cost-NO_EDGE"
        cost_robustness[ticker] = dict(survives_at=survives, verdict=verdict)
        print(f"    {ticker}: survives_at={survives} -> {verdict}")

    print("")
    print("  zero-cost edge-count check (should match "
          "momentum_lookback_sweep_results.json):")
    for ticker in ticker_order:
        e = sum(1 for lb in LOOKBACKS if
                lookback_verdict(results["zero"][lb][ticker]) == "EDGE")
        print(f"    {ticker}: {e}/3 lookbacks EDGE")

    # Determinism: rerun the AMZN/JPM subset under every cost level and assert
    # identical output.
    print("\n=== 6. Determinism (recompute AMZN/JPM under every cost level) ===")
    res2 = {
        name: {lb: {a: segment_result(a, tickers[a], lb, cfg)
                    for a in target}
               for lb in LOOKBACKS}
        for name, cfg in COST_LEVELS.items()
    }
    out1 = json.dumps(results, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    # Rebuild the same structure from the fresh subset so comparison is fair.
    res1_subset = {
        name: {lb: {a: results[name][lb][a] for a in target}
               for lb in LOOKBACKS}
        for name in COST_LEVELS
    }
    out1b = json.dumps(res1_subset, sort_keys=True)
    identical = out1b == out2
    print("  determinism r1 == r2 (AMZN/JPM across cost levels and lookbacks): "
          "{}".format("IDENTICAL" if identical else "DIFFERENT"))
    assert identical, "non-deterministic cost-sensitivity output"

    # --- build artifact ---
    def asset_summary():
        out = {}
        for ticker in ticker_order:
            lb_summaries = {}
            for lb in LOOKBACKS:
                zero_sr = results["zero"][lb][ticker]
                edge = ("EDGE" if (zero_sr["verdict"] == "REGIME_STABLE"
                                   and all(m > 0 for m in zero_sr["medians"]))
                        else ("LOSS" if zero_sr["verdict"] == "REGIME_STABLE_LOSS"
                              else "NO_EDGE"))
                lb_summaries[str(lb)] = {
                    name: dict(results[name][lb][ticker])
                    for name in COST_LEVELS
                }
                lb_summaries[str(lb)]["zero_cost_edge"] = edge
            edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
            for lb in LOOKBACKS:
                edge_counts[lb_summaries[str(lb)]["zero_cost_edge"]] += 1
            lb_summaries["edge_counts"] = edge_counts
            lb_summaries["robustness_verdict"] = (
                "ROBUST" if edge_counts["EDGE"] >= 2 else (
                    "SENSITIVE" if edge_counts["EDGE"] == 1 else "NO_EDGE"))
            out[ticker] = lb_summaries
        return out

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_grid=dict(
            (name, dict(
                slippage_proportional=cfg.slippage_proportional,
                commission_per_share=cfg.commission_per_share,
                commission_per_trade=cfg.commission_per_trade,
                slippage_cents=cfg.slippage_cents,
                initial_capital=cfg.initial_capital,
                target_exposure=cfg.target_exposure,
                ref_cost_per_notional=round(ref_cost_per_notional(cfg), 6),
            )) for name, cfg in COST_LEVELS.items()),
        cost_robustness=cost_robustness,
        per_asset=asset_summary(),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on AMZN/JPM + universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
