"""Cost-sensitivity check for short-horizon momentum (lookback 3 / 5 / 10).

Background: the momentum class (long the previous N-day return, hold 1 day,
daily rebalance) was admitted as candidate positive evidence on the collected
10-asset universe: 7/10 assets REGIME_STABLE with uniformly positive median
walk-forward log returns at lookback=5, robust across lookback 3/5/10 on real
data (state/check_artifacts/momentum_results.json,
state/check_artifacts/momentum_lookback_sweep_results.json). The outstanding
qualification is execution cost: momentum is a high-turnover mechanical rule
that rebalances daily, and the repository's existing realistic cost model may
consume the edge.

This check runs momentum lookback 3/5/10 across all 10 collected tickers under
a predeclared cost grid, with a matched coin-flip null computed at the same
cost level (noise_benchmark receives the same BacktestConfig), so turnover-
dependent costs drag both candidate and null while the relative edge is tested
against the cost-aware null.

Cost grid (fixed a-priori, not tuned to OOS results):
  zero_cost:     no transaction costs (reference).
  realistic:     the repository's existing realistic model used in
                 examples/ma_crossover.py and examples/ma_crossover_real_data.py:
                 commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005.
  conservative:  4x realistic (commission 8.0 / 0.012, slippage 8c / 0.002)
                 to stress robustness of a mechanical high-turnover rule.

Method: stress_segments_across_tickers / stress_segments / noise_benchmark on
the collected universe with the same walk-forward windows and regime family as
research/checks/momentum.py and research/checks/momentum_lookback_sweep.py;
per-segment median log return compared vs the matched null; regime-stability
verdicts aggregated across the universe. Leakage review on AMZN/JPM at realistic
costs. Exact realized cost accounting (commission + slippage) extracted from
engine fills. Determinism asserted.

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

# Predeclared cost grid. realistic = the repository's existing realistic cost
# model (examples/ma_crossover.py, examples/ma_crossover_real_data.py).
cost_configs = [
    dict(
        name="zero_cost",
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    ),
    dict(
        name="realistic",
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    ),
    dict(
        name="conservative",
        commission_per_trade=8.0,
        commission_per_share=0.012,
        slippage_cents=8.0,
        slippage_proportional=0.002,
    ),
]

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def cost_cfg(name, c):
    return bt.BacktestConfig(
        initial_capital=1e6,
        target_exposure=1.0,
        warmup_periods=WARM,
        commission_per_trade=c["commission_per_trade"],
        commission_per_share=c["commission_per_share"],
        slippage_cents=c["slippage_cents"],
        slippage_proportional=c["slippage_proportional"],
    )


def _segments_from_labels(labels, min_segment_bars):
    """Reconstruction of contiguous segments from a past-only label series,
    keeping runs with at least min_segment_bars bars. Mirrors the segmentation
    logic used inside stress_segments (independent local re-implementation)."""
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs


def per_asset_result(ticker, bars, lookback, cost_name, cost):
    """Run stress_segments for one ticker, lookback and cost config.
    stress_segments computes the candidate median and the noise (coin-flip)
    median with the SAME BacktestConfig, so the null is matched at these costs."""
    seg_fn = _seg_fn_from_bars(bars)
    sig = bt.momentum_signals(bars.closes_array(), lookback=lookback)
    if len(sig) != len(bars):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(bars)}")
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
        cfg=cost_cfg(cost_name, cost),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=res.candidate_dispersion,
        null_dispersion=res.null_dispersion,
        n_folds=res.n_folds,
        n_periods=res.n_periods,
        verdict=res.overall_verdict,
    )


def _seg_fn_from_bars(bars):
    """Build a segment_fn from one ticker's bars: compute the past-only
    volatility-block labels once, then return a simple O(1) lookup by bar
    index (mirrors research.backtest.segment_fn_from_labels)."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)

    def seg_fn(_closes, i):
        return labels[i]

    return seg_fn


def cost_accounting(tickers, lookback, cost):
    """Exact realized cost accounting for one lookback at one cost config:
    mean commission + slippage per OOS fold (USD) and mean number of OOS
    round trips per fold, averaged across segments of all 10 tickers. Costs
    are summed from the engine's recorded fills, so this is the framework's
    own accounting of the cost model, independent of the equity curve."""
    per_ticker = {}
    total_fold_cost = []
    total_round_trips = 0
    total_segs = 0
    for ticker, bars in tickers.items():
        closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
            [float(b.close) for b in bars], dtype=np.float64)
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        runs = _segments_from_labels(labels, MIN_SEGMENT_BARS)
        seg_bars = list(bars) if isinstance(bars, bt.BarSequence) else bars
        sig = bt.momentum_signals(closes, lookback=lookback)
        seg_costs = []
        seg_trips = 0
        for _, s, e in runs:
            seg_b = seg_bars[s:e]
            seg_s = sig[s:e]
            rf = bt.walk_forward(
                seg_b, seg_s, train_window=TRAIN, test_window=TEST,
                warmup=WARM, overlap_window=OVERLAP, cfg=cost_cfg(cost, cost))
            fold_costs = []
            trips = 0
            for fold in rf.folds:
                fills = fold.oos_fills
                cost = sum(float(f.commission) for f in fills) + sum(
                    float(f.shares) * (f.price - float(bar.close))
                    for f, bar in zip(fills, seg_b))
                if cost != 0.0 or fills:
                    fold_costs.append(float(cost))
                    if len(fills) >= 2:
                        trips += 1
            if fold_costs:
                seg_costs.append(np.mean(fold_costs))
                seg_trips += trips
        if seg_costs:
            per_ticker[ticker] = (round(float(np.mean(seg_costs)), 4), int(seg_trips))
            total_fold_cost.extend(seg_costs)
            total_round_trips += seg_trips
            total_segs += 1
    mean_cost = round(float(np.mean(total_fold_cost)), 4) if total_fold_cost else 0.0
    mean_trips = int(total_round_trips / total_segs) if total_segs else 0
    return per_ticker, mean_cost, mean_trips


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

    print("\n=== 3. Leakage review at realistic costs ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cost_cfg("realistic", cost_configs[1]))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        n_trades = len(res.trades)
        realized_cost = sum(float(f.commission) for f in res.trades) + sum(
            float(f.shares) * (f.price - float(bar.close)) for f, bar in zip(res.trades, bars))
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], trades={n_trades}, "
              f"total_return (realistic costs) {total_return:+.3f}, "
              f"realized commission+slippage ${realized_cost:,.0f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum cost sensitivity across collected universe ===")
    print("  Lookbacks: {}; cost tiers: zero_cost / realistic / conservative".format(LOOKBACKS))
    print("  Walk-forward per segment: train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 volatility blocks, each lookback "
          "vs a MATCHED coin-flip null computed at the same cost level".format(
          TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Full-universe cost sensitivity: lookback x cost tier.
    per_asset = {
        lb: {t: per_asset_result(t, all_tickers[t], lb, cc["name"], cc)
             for t in ticker_order}
        for lb in LOOKBACKS for cc in cost_configs
    }
    per_asset = {lb: {t: {c["name"]: per_asset[lb][(t, c["name"])] for c in cost_configs}
                      for t in ticker_order}
                 for lb in LOOKBACKS}

    # Verdict counts per cost tier, aggregated across lookbacks 3/5/10.
    def verdict_counts_for(tier):
        counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0,
                  "REGIME_STABLE_LOSS": 0, "REGIME_DEPENDENT": 0}
        for t in ticker_order:
            for lb in LOOKBACKS:
                counts[per_asset[lb][t][tier]["verdict"]] += 1
        return counts

    # Zero-cost cross-consistency vs the existing lookback sweep artifact.
    print("\n  Zero-cost reference (lookback 5) cross-check vs "
          "momentum_lookback_sweep_results.json:\n")
    with open(SWEEP_ARTIFACT) as f:
        expected = json.load(f)
    cmismatch = False
    for ticker in ticker_order:
        sw5 = per_asset[5][ticker]["zero_cost"]
        ep5 = expected["universe"]["per_asset"][ticker]
        ok = (sw5["medians"] == [round(v, 3) for v in ep5["lookback_medians"]["5"]]
              and sw5["null_medians"] == [round(v, 3) for v in ep5["lookback_null_medians"]["5"]]
              and sw5["verdict"] == ep5["verdicts_5"])
        if not ok:
            cmismatch = True
        print("    {}: zero-cost lookback5 medians {} vs {} | verdict {} -> {}".format(
            ticker, sw5["medians"], ep5["lookback_medians"]["5"],
            sw5["verdict"], "MATCH" if ok else "MISMATCH"))
    zero_check = "MATCH" if not cmismatch else "MISMATCH"
    if not cmismatch:
        print("    all 10 tickers cross-checked -> MATCH (zero-cost layer is an "
              "independent re-run of the existing lookback sweep)")

    # Verdict counts per cost tier.
    print("\n  Verdict counts (averaged over lookbacks 3/5/10):")
    for cc in cost_configs:
        name = cc["name"]
        cnt = verdict_counts_for(name)
        print("    {:12s}: {}".format(name, ", ".join("{}={}".format(k, v) for k, v in cnt.items())))

    # Cost accounting: exact realized costs from engine fills, one lookback at
    # realistic costs (the tier that matters for the decision), averaged over
    # all segments of all 10 tickers.
    print("\n  Realized cost accounting at realistic costs (engine fills, "
          "per OOS fold, averaged over all segments and all 10 tickers):")
    cost_summary = {}
    mean_cost_lb = {}
    mean_trips_lb = {}
    for lb in LOOKBACKS:
        per_ticker, mean_cost, mean_trips = cost_accounting(all_tickers, lb, cost_configs[1])
        cost_summary[str(lb)] = per_ticker
        mean_cost_lb[str(lb)] = mean_cost
        mean_trips_lb[str(lb)] = mean_trips
        print("    lookback {}: mean cost per OOS fold ${:.4f} (commission + "
              "slippage), mean {} round trips per fold".format(lb, mean_cost, mean_trips))

    # Summary print: lookback 5 medians and verdicts under each cost tier.
    print("\n  Lookback 5 (reference) by cost tier (median, verdict; null median in parens):")
    for cc in cost_configs:
        name = cc["name"]
        parts = []
        for ticker in ticker_order:
            r = per_asset[5][ticker][name]
            parts.append("{}:{:6.3f} {} ({:6.3f})".format(
                ticker, r["medians"][0], r["verdict"], r["null_medians"][0]))
        print("    [{}]".format(name))
        print("      " + " | ".join(parts))

    # Edge erosion: gap = candidate_median - null_median, delta vs zero cost.
    print("\n  Edge erosion (realistic vs zero cost): gap = candidate median - "
          "matched null median, delta per asset:")
    for ticker in ticker_order:
        gap0 = per_asset[5][ticker]["zero_cost"]["medians"][0] - per_asset[5][ticker]["zero_cost"]["null_medians"][0]
        gapr = per_asset[5][ticker]["realistic"]["medians"][0] - per_asset[5][ticker]["realistic"]["null_medians"][0]
        print("    {:6s}: zero-cost gap {:+.4f} -> realistic gap {:+.4f} "
              "(delta {:+.4f})".format(ticker, gap0, gapr, gapr - gap0))

    # Determinism: rerun the target subset and assert identical output.
    res2 = {
        lb: {t: per_asset_result(t, tickers[t], lb, cc["name"], cc)
             for t in target for cc in cost_configs}
    }
    res2 = {lb: {t: {c["name"]: res2[lb][(t, c["name"])] for c in cost_configs}
                 for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps(
        {lb: {t: {cn: per_asset[lb][t][cn] for cn in [cc["name"] for cc in cost_configs]}
               for t in target}
         for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_grid=[
            dict(name=cc["name"],
                 commission_per_trade=cc["commission_per_trade"],
                 commission_per_share=cc["commission_per_share"],
                 slippage_cents=cc["slippage_cents"],
                 slippage_proportional=cc["slippage_proportional"])
            for cc in cost_configs
        ],
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset={str(lb): per_asset[lb] for lb in LOOKBACKS},
        cost_accounting={"realistic": cost_summary},
        zero_cost_cross_check=zero_check,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost sensitivity across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
