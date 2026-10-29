"""Cost-sensitivity check for momentum lookback 3/5/10 under realistic transaction costs.

Objective: test whether the momentum edge (lookback 3 / 5 / 10) on the collected
universe survives a conservative transaction-cost model (commission + slippage).

Method (fixed a-priori, not tuned to OOS): long the previous lookback-day return;
hold 1 day; daily rebalance; lookbacks 3, 5, 10; walk-forward per volatility
segment with realistic costs; regime-stability verdict against a coin-flip null
matched for cost profile; deterministic across target assets.

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

# Realistic cost model based on examples/ma_crossover.py
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=60,
)

ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    warmup_periods=60,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sweep_results.json"


def segment_result(ticker, bars, lookback, cost_cfg):
    """Walk-forward regime-stability result for one ticker, lookback, and cost config."""
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
        cfg=cost_cfg,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def cost_sweep_verdict(sr_zero, sr_real):
    """Cost-robustness verdict for one lookback.

    CONSISTENT_WITH_NOISE_ZERO : indistinguishable from zero-cost null.
    CONSISTENT_WITH_NOISE_REAL  : indistinguishable from realistic-cost null.
    EDGE_ZERO_ONLY            : edge at zero cost but vanishes with realistic costs.
    EDGE_REAL_ONLY           : edge survives realistic costs (positive edge with realistic null).
    CONSISTENT_WITH_NOISE both conditions must hold (zero and realistic null).
    """
    zero_edge = sr_zero["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr_zero["medians"])
    real_edge = sr_real["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr_real["medians"])
    zero_noise = sr_zero["verdict"] == "CONSISTENT_WITH_NOISE"
    real_noise = sr_real["verdict"] == "CONSISTENT_WITH_NOISE"

    if zero_noise and real_noise:
        return "CONSISTENT_WITH_NOISE"
    if zero_edge and real_edge:
        return "EDGE_REAL_ONLY"
    if zero_edge and not real_edge:
        return "EDGE_ZERO_ONLY"
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

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Leakage review per asset (lookback=5 reference) ===")
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

    print("\n=== 4. Momentum lookback sweep under zero vs realistic costs ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "matched for cost profile\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep under both cost regimes
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][cost][segment index]
    results = {lb: {t: {"zero": None, "real": None} for t in ticker_order} for lb in LOOKBACKS}

    print("  Computing results under zero costs...")
    for lb in LOOKBACKS:
        for ticker in ticker_order:
            results[lb][ticker]["zero"] = segment_result(ticker, all_tickers[ticker], lb, ZERO_COSTS)

    print("  Computing results under realistic costs...")
    for lb in LOOKBACKS:
        for ticker in ticker_order:
            results[lb][ticker]["real"] = segment_result(ticker, all_tickers[ticker], lb, REALISTIC_COSTS)

    print("\n=== 5. Cost-robustness verdict counts across universe ===")
    per_asset = {}
    cost_counts = {"CONSISTENT_WITH_NOISE": 0, "EDGE_REAL_ONLY": 0, "EDGE_ZERO_ONLY": 0, "NO_EDGE": 0}

    for ticker in ticker_order:
        profile = {str(lb): {"zero": results[lb][ticker]["zero"], "real": results[lb][ticker]["real"]} for lb in LOOKBACKS}
        # Build verdict by lookback
        verdict_by_lookback = {}
        for lb in LOOKBACKS:
            verdict_by_lookback[str(lb)] = cost_sweep_verdict(
                results[lb][ticker]["zero"], results[lb][ticker]["real"])
        cost_counts[verdict_by_lookback["5"]] += 1
        profile["cost_verdicts"] = verdict_by_lookback
        per_asset[ticker] = profile

    print("  Cost-robustness verdict counts:")
    print("    CONSISTENT_WITH_NOISE = edge indistinguishable from both nulls")
    print("    EDGE_REAL_ONLY = positive edge survives realistic costs (strong evidence)")
    print("    EDGE_ZERO_ONLY = edge vanishes under realistic costs (weak evidence)")
    print("    NO_EDGE = no edge in either regime")
    for k, v in cost_counts.items():
        print(f"    {k}={v}")

    print("\n  Per-asset cost profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        zero_verdicts = [f"{lb}:{p['cost_verdicts'][str(lb)]}" for lb in LOOKBACKS]
        print(f"    {ticker:6s} zero_costs: {' '.join(zero_verdicts)}")

    # Determinism: rerun the target subset and assert identical output
    res2 = {lb: {t: {"zero": segment_result(t, tickers[t], lb, ZERO_COSTS),
                      "real": segment_result(t, tickers[t], lb, REALISTIC_COSTS)}
                 for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        zero_cost_config={
            "commission_per_trade": ZERO_COSTS.commission_per_trade,
            "commission_per_share": ZERO_COSTS.commission_per_share,
            "slippage_cents": ZERO_COSTS.slippage_cents,
            "slippage_proportional": ZERO_COSTS.slippage_proportional,
        },
        realistic_cost_config={
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        per_asset=per_asset,
        universe=dict(
            cost_counts=cost_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    zero_medians={str(lb): per_asset[t][str(lb)]["zero"]["medians"]
                                  for lb in LOOKBACKS},
                    zero_null_medians={str(lb): per_asset[t][str(lb)]["zero"]["null_medians"]
                                       for lb in LOOKBACKS},
                    real_medians={str(lb): per_asset[t][str(lb)]["real"]["medians"]
                                  for lb in LOOKBACKS},
                    real_null_medians={str(lb): per_asset[t][str(lb)]["real"]["null_medians"]
                                       for lb in LOOKBACKS},
                    zero_verdict={str(lb): per_asset[t][str(lb)]["cost_verdicts"][str(lb)]
                                  for lb in LOOKBACKS},
                    real_verdict={str(lb): per_asset[t][str(lb)]["cost_verdicts"][str(lb)]
                                  for lb in LOOKBACKS},
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check across the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
