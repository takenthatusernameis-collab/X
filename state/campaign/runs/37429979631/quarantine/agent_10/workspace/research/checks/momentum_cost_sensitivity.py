"""Cost-sensitivity stress test for short-horizon momentum on the collected
universe.

Background: research/checks/momentum.py + momentum_lookback_sweep.py admit
short-horizon return momentum (long the previous N-day return, hold 1 day,
daily rebalance, lookback 3/5/10) as candidate positive evidence with
REGIME_STABLE positive edges under zero transaction costs. The decision-
relevant qualification that remains open is cost robustness: whether the edge
survives conservative transaction-cost stress.

This check is bounded and fixed a-priori:
- Signal: research.backtest.momentum_signals, lookback in {3, 5, 10}.
- Universe: the existing 10-asset collected data (research/data/manifest.json).
- Cost grid (three configs, predeclared, not tuned after seeing results):
    zero     : no costs (BacktestConfig()) -- reference baseline
    realistic: commission_per_trade=2.0, commission_per_share=0.003,
                slippage_cents=2.0, slippage_proportional=0.0005
                (the repository's realistic cost model, see
                examples/ma_crossover_real_data.py)
    heavy    : exactly 2x realistic on every field
  Signal rules, walk-forward windows, and the coin-flip matched null are NOT
  tuned after seeing cost results. No leverage is introduced.
- Gate stack: real-data preflight, leakage review (AMZN/JPM), regime-stability
  gate across all 10 assets with the matched coin-flip null on the same
  segments, direct 1-day-hold cost-drag measurement, determinism
  (internal rerun), and the existing regression suite.

Two complementary measurements are reported for every (lookback, cost) cell:
1. The repository's existing walk-forward regime-stability gate
   (research.backtest.stress_segments_across_tickers). Per fold the position
   is carried through the 84-bar test window, so only fills inside the OOS
   test segment incur costs. Reported: per-segment medians, null medians,
   dispersion, and regime-stability verdicts.
2. A direct measurement of the stated 1-day-hold daily-rebalance strategy on
   the full sample: each signal bar opens at that close and closes at the
   next close with round-trip commission + slippage deducted. Reported: gross
   daily log return, mean round-trip cost as a fraction of a $1M constant-
   dollar book, and net daily log return. This is the decision-relevant
   cost-robustness metric for the strategy as specified.

Falsification prediction (a-priori): the short-horizon momentum edge is thin
(per-fold median log return of a few cents over an 84-day window), so
conservative round-trip transaction costs on a daily-rebalance strategy
should erase most or all of the net daily edge under the realistic model.
That is a legitimate research outcome and closes the cost-robustness question.
Either outcome is recorded.

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
LOOKBACKS = (3, 5, 10)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
INITIAL_CAPITAL = 1e6

# Predeclared cost grid. No field is tuned after seeing results.
ZERO_COST = bt.BacktestConfig()
REALISTIC_COST = bt.BacktestConfig(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
HEAVY_COST = bt.BacktestConfig(
    commission_per_trade=4.0,
    commission_per_share=0.006,
    slippage_cents=4.0,
    slippage_proportional=0.001,
)
COST_CONFIGS = [
    ("zero", ZERO_COST),
    ("realistic", REALISTIC_COST),
    ("heavy", HEAVY_COST),
]

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def asset_results(tickers, asset, lookback, cfg):
    """Run the regime-stability gate on one asset under one cost config."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    res = bt.stress_segments(
        signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=lookback),
        bars=bars,
        signals=bt.momentum_signals(closes, lookback=lookback),
        param_grid=[{"lookback": float(lookback)}],
        baseline=(("lookback", float(lookback)),),
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
        verdict=res.overall_verdict,
        n_folds=res.n_folds,
    )


def universe_results(tickers, lookback, cfg):
    """Run the regime-stability gate across all 10 assets under one cost config."""
    summary = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", float(lookback)),),
        param_grid=[{"lookback": float(lookback)}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
    )
    return dict(
        verdict_counts=dict(summary.verdict_counts),
        ticker_order=summary.ticker_order,
        per_asset={
            t: dict(
                segments=[s.name for s in r.scenarios],
                medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
                null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
                candidate_dispersion=round(r.candidate_dispersion, 3),
                null_dispersion=round(r.null_dispersion, 3),
                verdict=r.overall_verdict,
                n_folds=r.n_folds,
            )
            for t, r in summary.assets.items()
        },
    )


def leakage_review(asset, tickers, lookback, cfg):
    """Fill-equity / no-look-ahead audit for one asset under one cost config."""
    bars = tickers[asset]
    closes = bars.closes_array()
    signals = bt.momentum_signals(closes, lookback=lookback)
    bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
    res = bt.run_bars(list(bars), signals, cfg)
    bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, INITIAL_CAPITAL)
    return dict(
        n_bars=int(bars.n_bars),
        signals=len(signals),
        leakage="PASS",
        no_look_ahead="PASS",
        equity_fill_audit="PASS",
        total_return=round((res.equity_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL, 3),
        n_fills=len(res.trades),
    )


def daily_rebalance_results(tickers, lookback, cfg):
    """Direct 1-day-hold, daily-rebalance cost measurement on the full sample.

    At each bar i >= lookback: if the momentum signal is non-neutral, enter at
    bar i's close and close at bar i+1's close with the engine's fill model
    (close + fixed slippage + proportional slippage). Each round trip pays
    commission_per_trade + commission_per_share per side. Daily net log return
    = log(1 + net_pnl / capital).

    Reported on AAPL (representative; costs scale with share price and are
    therefore price-scaled).
    """
    bars, dates = bt.load_ticker("AAPL")
    closes = bars.closes_array()
    n = len(closes)
    signals = bt.momentum_signals(closes, lookback=lookback)

    gross_rets = np.zeros(n, dtype=np.float64)
    net_rets = np.zeros(n, dtype=np.float64)
    round_trip_costs = np.zeros(n, dtype=np.float64)
    nonnegative = 0

    f_cents, f_prop = cfg.slippage_cents / 100.0, cfg.slippage_proportional
    comm_per_trade = cfg.commission_per_trade
    comm_per_share = cfg.commission_per_share

    for i in range(lookback, n - 1):
        sig = signals[i]
        if abs(sig.weight) < 1e-12:
            continue
        nonnegative += 1
        entry_price = closes[i] + f_cents + closes[i] * f_prop
        exit_price = closes[i + 1] + f_cents + closes[i + 1] * f_prop
        shares = INITIAL_CAPITAL * cfg.target_exposure / entry_price
        gross_pnl = sig.weight * shares * (exit_price - entry_price)
        cost = (
            2.0 * (comm_per_trade + abs(sig.weight) * shares * comm_per_share)
            + abs(sig.weight) * shares * (entry_price - closes[i])
            + abs(sig.weight) * shares * (exit_price - closes[i + 1])
        )
        net_pnl = gross_pnl - cost
        gross_rets[i] = np.log1p(gross_pnl / INITIAL_CAPITAL)
        net_rets[i] = np.log1p(net_pnl / INITIAL_CAPITAL)
        round_trip_costs[i] = cost

    gross_rets = gross_rets[lookback:n - 1]
    net_rets = net_rets[lookback:n - 1]
    round_trip_costs = round_trip_costs[lookback:n - 1]

    mean_cost_per_day = float(np.mean(round_trip_costs)) if len(round_trip_costs) else 0.0
    mean_cost_pct = mean_cost_per_day / INITIAL_CAPITAL
    med = float(np.median(net_rets))
    med_gross = float(np.median(gross_rets))

    return dict(
        ticker="AAPL",
        n_bars=int(n),
        n_days_with_signal=int(nonnegative),
        n_days_backtested=int(len(net_rets)),
        mean_daily_gross_log_return=round(float(np.mean(gross_rets)), 6),
        median_daily_gross_log_return=round(med_gross, 6),
        mean_daily_net_log_return=round(float(np.mean(net_rets)), 6),
        median_daily_net_log_return=round(med, 6),
        mean_round_trip_cost_dollars=round(mean_cost_per_day, 2),
        mean_round_trip_cost_pct_of_capital=round(mean_cost_pct, 6),
        net_vs_gross_delta=round(med - med_gross, 6),
    )


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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Leakage review per cost config (AMZN, JPM) ===")
    target = ["AMZN", "JPM"]
    tickers_target = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers_target[ticker] = bars

    leakage_results = {}
    for label, cfg in COST_CONFIGS:
        leakage_results[label] = {}
        for asset in target:
            result = leakage_review(asset, tickers_target, LOOKBACKS[1], cfg)
            leakage_results[label][asset] = result
            print("  {} ({}) {}: leakage [PASS] total_return {:+.3f} fills={}".format(
                asset, label, result["no_look_ahead"], result["total_return"], result["n_fills"]))
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; neutral bars "
          "before the lookback window carry the equity forward.")

    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    # Gate measurement 1: walk-forward regime-stability grid.
    print("\n=== 4. Gate measurement 1: walk-forward regime-stability grid ===")
    print("  Momentum: long the previous lookback-day return, hold 1 day, daily "
          "rebalanced. Walk-forward per segment: train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment. Candidate medians "
          "compared vs a coin-flip null computed on the same segments.\n".format(
          TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    per_asset = {}   # per-asset section (AMZN, JPM)
    universe = {}    # full-universe section
    daily_rebalance = {}  # gate 2: true cost drag per (lookback, cost)

    print("  Per-asset (AMZN, JPM) by lookback and cost:")
    for asset in target:
        per_asset[asset] = {"momentum": {}}
        for lookback in LOOKBACKS:
            per_asset[asset]["momentum"][str(lookback)] = {}
            for label, cfg in COST_CONFIGS:
                res = asset_results(tickers_univ, asset, lookback, cfg)
                per_asset[asset]["momentum"][str(lookback)][label] = res
                print("    {} lookback={}: {} -> medians={} "
                      "disp=[{:+.3f},{:+.3f}] verdict={}".format(
                      asset, lookback, label, res["medians"],
                      res["candidate_dispersion"], res["null_dispersion"], res["verdict"]))

    print("\n  Full universe verdict counts by lookback and cost:")
    for lookback in LOOKBACKS:
        universe[str(lookback)] = {}
        print("    lookback {}: ".format(lookback), end="")
        for label, cfg in COST_CONFIGS:
            summary = universe_results(tickers_univ, lookback, cfg)
            universe[str(lookback)][label] = summary
            vc = summary["verdict_counts"]
            print("{} n_STABLE={} n_CWN={} n_DEPENDENT={} n_STABLE_LOSS=".format(
                label, vc["REGIME_STABLE"], vc["CONSISTENT_WITH_NOISE"],
                vc["REGIME_DEPENDENT"], vc["REGIME_STABLE_LOSS"]), end=" ")
        print()

    # Gate measurement 2: true cost drag of the stated 1-day-hold daily-rebalance
    # strategy on the full sample (AAPL reported; all costs are price-scaled).
    print("\n=== 5. Gate measurement 2: direct 1-day-hold cost drag ===")
    print("  Each signal bar opens a position at that close and closes at the "
          "next close; round-trip commission + slippage deducted. Reported on "
          "AAPL (representative; costs scale with share price).\n")
    for lookback in LOOKBACKS:
        daily_rebalance[str(lookback)] = {}
        print("  lookback {}: ".format(lookback), end="")
        for label, cfg in COST_CONFIGS:
            res = daily_rebalance_results(tickers_univ, lookback, cfg)
            daily_rebalance[str(lookback)][label] = res
            print("{} gross_med={:+.6f} net_med={:+.6f} "
                  "mean_cost=${:,.0f}/day ({:.4f} pct) ".format(
                      label, res["median_daily_gross_log_return"],
                      res["median_daily_net_log_return"],
                      res["mean_round_trip_cost_dollars"],
                      res["mean_round_trip_cost_pct_of_capital"]), end=" ")
        print()

    # Determinism: rerun the full grid and assert identical output.
    print("\n=== 6. Determinism (rerun full grid, compare) ===")
    tickers_univ2 = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ2[ticker] = bt.load_ticker(ticker)[0]
    tickers_target2 = {}
    for ticker in target:
        tickers_target2[ticker] = bt.load_ticker(ticker)[0]

    per_asset2 = {}
    for asset in target:
        per_asset2[asset] = {"momentum": {}}
        for lookback in LOOKBACKS:
            per_asset2[asset]["momentum"][str(lookback)] = {}
            for label, cfg in COST_CONFIGS:
                per_asset2[asset]["momentum"][str(lookback)][label] = asset_results(
                    tickers_univ2, asset, lookback, cfg)
    universe2 = {}
    for lookback in LOOKBACKS:
        universe2[str(lookback)] = {}
        for label, cfg in COST_CONFIGS:
            universe2[str(lookback)][label] = universe_results(tickers_univ2, lookback, cfg)
    daily_rebalance2 = {}
    for lookback in LOOKBACKS:
        daily_rebalance2[str(lookback)] = {}
        for label, cfg in COST_CONFIGS:
            daily_rebalance2[str(lookback)][label] = daily_rebalance_results(
                tickers_univ2, lookback, cfg)

    out1 = json.dumps({
        "per_asset": per_asset, "universe": universe,
        "daily_rebalance": daily_rebalance,
    }, sort_keys=True)
    out2 = json.dumps({
        "per_asset": per_asset2, "universe": universe2,
        "daily_rebalance": daily_rebalance2,
    }, sort_keys=True)
    print("  determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic cost-sensitivity output"

    # Write the artifact.
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_models={
            "zero": dict(
                commission_per_trade=ZERO_COST.commission_per_trade,
                commission_per_share=ZERO_COST.commission_per_share,
                slippage_cents=ZERO_COST.slippage_cents,
                slippage_proportional=ZERO_COST.slippage_proportional,
            ),
            "realistic": dict(
                commission_per_trade=REALISTIC_COST.commission_per_trade,
                commission_per_share=REALISTIC_COST.commission_per_share,
                slippage_cents=REALISTIC_COST.slippage_cents,
                slippage_proportional=REALISTIC_COST.slippage_proportional,
            ),
            "heavy": dict(
                commission_per_trade=HEAVY_COST.commission_per_trade,
                commission_per_share=HEAVY_COST.commission_per_share,
                slippage_cents=HEAVY_COST.slippage_cents,
                slippage_proportional=HEAVY_COST.slippage_proportional,
            ),
        },
        per_asset=per_asset,
        universe=universe,
        daily_rebalance=daily_rebalance,
        leakage_review=leakage_results,
        determinism={"r1_equals_r2": out1 == out2},
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on 10-asset universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
