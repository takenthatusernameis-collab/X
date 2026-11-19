"""Bounded transaction-cost sensitivity check for short-horizon momentum.

Objective: classify whether the corrected lookback 3/5/10 momentum signal
survives conservative transaction-cost stress on the collected real-data
universe, using the repository's existing cost model and matched coin-flip null.

The cost grid is fixed before execution:
- zero: no commission, no per-share commission, no slippage;
- realistic: the repository's existing example cost model
  (commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005);
- conservative: exactly twice every realistic cost component.

The signal is the corrected short-horizon momentum rule: sign of
log(close[i] / close[i-lookback]), held one day, daily rebalance. Each
asset/lookback/cost cell is evaluated with the same walk-forward and volatility
segments as the existing momentum checks, and the matched coin-flip null uses
the same cost configuration.

Predeclared classification:
- EDGE: REGIME_STABLE and every segment median > 0.
- An asset is lookback-cost-robust at a cost level if EDGE occurs at >= 2 of
  the three lookbacks.
- COST_ROBUST asset: zero, realistic, and conservative all lookback-cost-robust.
- COST_FRAGILE asset: zero is lookback-cost-robust but realistic or
  conservative is not.
- NO_BASELINE_EDGE asset: zero is not lookback-cost-robust.
- Universe COST_ROBUST if >= 6/10 assets are COST_ROBUST; otherwise
  COST_FRAGILE if >= 6/10 assets have a zero-cost lookback-robust edge;
  NO_BASELINE_EDGE if no asset has a zero-cost lookback-robust edge; otherwise
  MIXED.

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

COST_CONFIGS = {
    "zero": dict(
        commission_per_trade=0.0,
        commission_per_share=0.0,
        slippage_cents=0.0,
        slippage_proportional=0.0,
    ),
    "realistic": dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    ),
    "conservative": dict(
        commission_per_trade=4.0,
        commission_per_share=0.006,
        slippage_cents=4.0,
        slippage_proportional=0.0010,
    ),
}
COST_ORDER = ("zero", "realistic", "conservative")

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def segment_result(ticker, bars, lookback, cost_label):
    """One asset/lookback/cost walk-forward regime-stability result."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    cfg = bt.BacktestConfig(**COST_CONFIGS[cost_label])
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


def lookback_verdict(result):
    """EDGE / LOSS / NO_EDGE for one asset/lookback/cost cell."""
    if result["verdict"] == "REGIME_STABLE" and all(m > 0 for m in result["medians"]):
        return "EDGE"
    if result["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def classify_asset(cost_results):
    """Predeclared asset-level cost-robustness classification."""
    edge_counts = {}
    for cost_label in COST_ORDER:
        edge_counts[cost_label] = sum(
            1 for lb in LOOKBACKS
            if lookback_verdict(cost_results[cost_label][str(lb)]) == "EDGE"
        )
    if edge_counts["zero"] < 2:
        label = "NO_BASELINE_EDGE"
    elif edge_counts["realistic"] >= 2 and edge_counts["conservative"] >= 2:
        label = "COST_ROBUST"
    else:
        label = "COST_FRAGILE"
    return label, edge_counts


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
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(
            **COST_CONFIGS["realistic"]))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], realistic-cost total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes up to t only; "
          "neutral bars before the lookback window carry equity forward.")

    print("\n=== 4. Predeclared cost grid ===")
    for cost_label in COST_ORDER:
        print(f"  {cost_label}: {COST_CONFIGS[cost_label]}")
    print("  lookbacks: 3 / 5 / 10; train={}d / test={}d / warmup={}d / "
          "overlap={}d; min {} bars/segment; 4 volatility blocks".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("  null: matched coin-flip benchmark under the same cost configuration")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    per_asset = {}
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        cost_results = {
            cost_label: {
                str(lb): segment_result(ticker, bars, lb, cost_label)
                for lb in LOOKBACKS
            }
            for cost_label in COST_ORDER
        }
        label, edge_counts = classify_asset(cost_results)
        per_asset[ticker] = {
            "costs": cost_results,
            "edge_counts": edge_counts,
            "cost_robustness": label,
        }

    counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0,
              "NO_BASELINE_EDGE": 0}
    for ticker in ticker_order:
        counts[per_asset[ticker]["cost_robustness"]] += 1
    zero_robust = sum(
        1 for ticker in ticker_order
        if per_asset[ticker]["edge_counts"]["zero"] >= 2
    )
    if counts["COST_ROBUST"] >= 6:
        universe_label = "COST_ROBUST"
    elif zero_robust >= 6:
        universe_label = "COST_FRAGILE"
    elif zero_robust == 0:
        universe_label = "NO_BASELINE_EDGE"
    else:
        universe_label = "MIXED"

    verdict_counts = {}
    for cost_label in COST_ORDER:
        for lb in LOOKBACKS:
            c = {"REGIME_STABLE": 0, "REGIME_DEPENDENT": 0,
                 "CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE_LOSS": 0}
            for ticker in ticker_order:
                c[per_asset[ticker]["costs"][cost_label][str(lb)]["verdict"]] += 1
            verdict_counts[f"{cost_label}_lookback_{lb}"] = c

    print("\n=== 5. Per-asset cost profiles ===")
    for ticker in ticker_order:
        p = per_asset[ticker]
        fields = []
        for cost_label in COST_ORDER:
            meds = []
            for lb in LOOKBACKS:
                meds.append(
                    "{}:{:+.3f} {}".format(
                        lb,
                        p["costs"][cost_label][str(lb)]["medians"][0],
                        lookback_verdict(p["costs"][cost_label][str(lb)]),
                    )
                )
            fields.append("{} edges={} [{}]".format(
                cost_label, p["edge_counts"][cost_label], "; ".join(meds)))
        print("  {:6s} {:16s} | {}".format(
            ticker, p["cost_robustness"], " | ".join(fields)))
    print("")

    print("=== 6. Universe classification ===")
    print(f"  zero-cost lookback-robust assets: {zero_robust}/10")
    print(f"  asset counts: {counts}")
    print(f"  universe cost-robustness: {universe_label}")
    print("  verdict counts by cost and lookback:")
    for key, c in verdict_counts.items():
        print(f"    {key}: {c}")
    print("")

    determinism_target = {}
    for ticker in target:
        for cost_label in COST_ORDER:
            for lb in LOOKBACKS:
                determinism_target[(ticker, cost_label, lb)] = segment_result(
                    ticker, tickers[ticker], lb, cost_label)
    out1 = json.dumps({
        ticker: {
            cost_label: {
                str(lb): per_asset[ticker]["costs"][cost_label][str(lb)]
                for lb in LOOKBACKS
            }
            for cost_label in COST_ORDER
        }
        for ticker in target
    }, sort_keys=True)
    out2 = json.dumps({
        ticker: {
            cost_label: {
                str(lb): determinism_target[(ticker, cost_label, lb)]
                for lb in LOOKBACKS
            }
            for cost_label in COST_ORDER
        }
        for ticker in target
    }, sort_keys=True)
    deterministic = out1 == out2
    print("determinism: r1 == r2: {}".format(deterministic))
    assert deterministic, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        signal=dict(
            name="momentum",
            formula="sign(log(close[i] / close[i-lookback]))",
            lookbacks=list(LOOKBACKS),
        ),
        cost_grid={label: COST_CONFIGS[label] for label in COST_ORDER},
        classification_rule=dict(
            edge="REGIME_STABLE and all segment medians > 0",
            lookback_cost_robust="EDGE at >= 2 of 3 lookbacks",
            cost_robust_asset=(
                "zero, realistic, and conservative are all "
                "lookback-cost-robust"
            ),
            cost_fragile_asset=(
                "zero is lookback-cost-robust but realistic or "
                "conservative is not"
            ),
            no_baseline_edge_asset="zero is not lookback-cost-robust",
            universe=(
                "COST_ROBUST if >= 6/10 COST_ROBUST assets; else "
                "COST_FRAGILE if >= 6/10 zero-cost lookback-robust assets; "
                "else NO_BASELINE_EDGE if none; else MIXED"
            ),
        ),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            ticker_order=ticker_order,
            counts=counts,
            zero_cost_lookback_robust=zero_robust,
            classification=universe_label,
            verdict_counts=verdict_counts,
            per_asset={
                ticker: dict(
                    edge_counts=per_asset[ticker]["edge_counts"],
                    cost_robustness=per_asset[ticker]["cost_robustness"],
                )
                for ticker in ticker_order
            },
        ),
        determinism=dict(
            checked=True,
            r1_equals_r2=deterministic,
            target=list(target),
            costs=list(COST_ORDER),
            lookbacks=list(LOOKBACKS),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check on the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())