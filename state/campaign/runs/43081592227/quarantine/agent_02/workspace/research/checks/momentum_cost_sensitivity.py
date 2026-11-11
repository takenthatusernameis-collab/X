"""Real-data transaction-cost sensitivity of short-horizon momentum.

Objective (fixed a-priori, not tuned to results): the qualified momentum
signal (long the previous N-day return, hold 1 day, daily rebalance) was
admitted as candidate positive evidence at zero cost on the collected
universe. This check asks whether that edge survives conservative
transaction-cost stress. Momentum is a high-turnover, daily-rebalanced
signal, so execution cost is the decision-relevant qualification.

Cost grid (predeclared before execution; identical for candidate and null):

- ``zero``        : the framework default (no commission, no slippage). This
                    is a reference column only; it must reproduce
                    ``momentum_lookback_sweep_results.json`` exactly.
- ``realistic``   : the repository's existing realistic cost model
                    (commission $2.00/trade + $0.003/share, 2c fixed slippage,
                    5 bps proportional slippage), the same model used by
                    ``examples/ma_crossover.py`` and
                    ``examples/ma_crossover_real_data.py``.
- ``conservative``: 2x the realistic model (commission $4.00/trade +
                    $0.006/share, 4c fixed slippage, 10 bps proportional), a
                    conservative stress level for a high-turnover signal.

Method: for each lookback in {3, 5, 10} and each of the 10 collected tickers,
momentum is walk-forward validated (train=252d/test=84d/warmup=60d/overlap=60d)
inside each contiguous volatility block (past-only ``volatility_blocks``), using
the framework's ``stress_segments`` with the cost model applied. The candidate's
per-segment median log return is compared against a coin-flip sign null run on
the same segment under the same cost model (the matched null). Each segment's
cost-model effect is therefore measured apples-to-apples.

A-priori prediction (declared before execution): the edge is expected to be
cost-sensitive but not uniformly erased. The strongest, most persistent assets
(AAPL, MSFT, GOOGL, AMZN, META, TSLA) are expected to remain positive at the
realistic model and to degrade (but stay positive, or turn marginal) at the
conservative model; the marginal assets (JNJ, XOM) are expected to be pushed to
non-positive at the realistic or conservative level. If the realistic model
erases the positive median across most of the qualified set, the momentum
admission must be downgraded to cost-fragile.

Classification (predeclared, per ticker, at a given lookback; based on the mean
over segments of the candidate median log return):

- NO_ZERO_COST_EDGE : no positive mean candidate median at zero cost.
- COST_ROBUST       : positive at both realistic and conservative cost.
- COST_DEGRADED     : positive at realistic cost but not at conservative cost.
- COST_ERASED       : non-positive at realistic cost.

The mean gap over the matched null is reported alongside for transparency but
does not drive the class, which is about absolute edge survival.

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
        "slippage_proportional": 0.001,
    },
}


def cost_cfg(level):
    return bt.BacktestConfig(**COST_LEVELS[level])


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker/lookback/cost level."""
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


def segment_stats(sr):
    """Mean candidate median, mean candidate-minus-null gap, positive segment count."""
    cand = [float(m) for m in sr["medians"]]
    null = [float(m) for m in sr["null_medians"]]
    gaps = [c - n for c, n in zip(cand, null)]
    return dict(
        mean_median=round(float(np.mean(cand)), 3),
        mean_gap=round(float(np.mean(gaps)), 3),
        pos_segments=int(sum(1 for m in cand if m > 0)),
        n_segments=len(cand),
    )


def classify(zero_stats, real_stats, cons_stats):
    """Predeclared cost-robustness class for one ticker/lookback."""
    if zero_stats["mean_median"] <= 0:
        return "NO_ZERO_COST_EDGE"
    if real_stats["mean_median"] > 0 and cons_stats["mean_median"] > 0:
        return "COST_ROBUST"
    if real_stats["mean_median"] > 0:
        return "COST_DEGRADED"
    return "COST_ERASED"


def run_all(tickers):
    """Run the full cost x lookback x ticker grid."""
    per_cost = {}
    for level in COST_LEVELS:
        cfg = cost_cfg(level)
        per_cost[level] = {
            str(lb): {t: segment_result(t, tickers[t], lb, cfg) for t in tickers}
            for lb in LOOKBACKS
        }
    return per_cost


def build_summary(per_cost):
    """Per-lookback, per-ticker cost stats and classification."""
    summary = {}
    class_counts = {}
    for lb in LOOKBACKS:
        key = str(lb)
        summary[key] = {}
        counts = {"COST_ROBUST": 0, "COST_DEGRADED": 0, "COST_ERASED": 0,
                  "NO_ZERO_COST_EDGE": 0}
        for t in per_cost["zero"][key]:
            z = segment_stats(per_cost["zero"][key][t])
            r = segment_stats(per_cost["realistic"][key][t])
            c = segment_stats(per_cost["conservative"][key][t])
            cls = classify(z, r, c)
            counts[cls] += 1
            summary[key][t] = dict(
                zero=z,
                realistic=r,
                conservative=c,
                zero_cost_edge=bool(z["mean_median"] > 0),
                beats_null_realistic=bool(r["mean_gap"] > 0),
                beats_null_conservative=bool(c["mean_gap"] > 0),
                cost_class=cls,
            )
        class_counts[key] = counts
    return summary, class_counts


def cost_model_control(tickers):
    """Expose the engine's cost convention with non-signal controls.

    A genuine cost cannot increase a strategy's total return. The engine prices
    every fill at ``close + slippage`` for both buys and sells, so a net-selling
    (long-biased) strategy is *credited* the slippage on its trims. This control
    runs a constant-long position and a fixed coin-flip signal through each cost
    level so that the convention is auditable alongside the candidate results.
    """
    out = {"constant_long": {}, "random_aapl": {}}
    for level in COST_LEVELS:
        cfg = cost_cfg(level)
        per_t = {}
        for t, bars in tickers.items():
            n = bars.n_bars
            const_long = [bt.Signal(date=i + 1, weight=1.0) for i in range(n)]
            res = bt.run_bars(list(bars), const_long, cfg)
            ret = float(res.equity_curve[-1] / res.equity_curve[0] - 1.0)
            per_t[t] = dict(
                const_long_total_return=round(ret, 6),
                net_shares=round(float(sum(f.shares for f in res.trades)), 1),
                gross_shares=round(float(sum(abs(f.shares) for f in res.trades)), 1),
                commission=round(float(sum(f.commission for f in res.trades)), 2),
            )
        out["constant_long"][level] = per_t
        bars = tickers["AAPL"]
        rsig = bt.random_signals(bars.n_bars, seed=5042)
        rres = bt.run_bars(list(bars), rsig, cfg)
        out["random_aapl"][level] = dict(
            random_total_return=round(
                float(rres.equity_curve[-1] / rres.equity_curve[0] - 1.0), 6),
            commission=round(float(sum(f.commission for f in rres.trades)), 2),
        )
    return out


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

    print("\n=== 3. Leakage review per asset (lookback=5, zero cost) ===")
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
          "costs are applied identically to candidate and matched null.")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    print("\n=== 4. Cost x lookback x ticker grid ===")
    print("  cost levels: " + ", ".join(COST_LEVELS))
    print("  realistic = commission $2/trade + $0.003/share, 2c fixed + 5bps prop")
    print("  conservative = 2x realistic\n")

    per_cost = run_all(all_tickers)
    summary, class_counts = build_summary(per_cost)
    control = cost_model_control(all_tickers)

    for lb in LOOKBACKS:
        key = str(lb)
        print("  lookback {}: {}".format(lb, class_counts[key]))
    print("")

    print("  Per-asset lookback-5 cost profile (mean segment median; class):")
    for t in ticker_order:
        s = summary["5"][t]
        print("    {:6s} zero={:+.3f} real={:+.3f} cons={:+.3f} "
              "gap_real={:+.3f} gap_cons={:+.3f} pos_seg={}/{} -> {}".format(
                  t, s["zero"]["mean_median"], s["realistic"]["mean_median"],
                  s["conservative"]["mean_median"], s["realistic"]["mean_gap"],
                  s["conservative"]["mean_gap"], s["realistic"]["pos_segments"],
                  s["realistic"]["n_segments"], s["cost_class"]))
    print("")

    n_ctrl_up = sum(
        1 for t in ticker_order
        if control["constant_long"]["realistic"][t]["const_long_total_return"]
        > control["constant_long"]["zero"][t]["const_long_total_return"]
    )
    print("  cost-model control (constant-long): return rises from zero to "
          "realistic cost for {}/{} tickers (a genuine cost cannot raise "
          "return; >0 flags the symmetric-slippage convention)".format(
              n_ctrl_up, len(ticker_order)))
    print("")
    res2 = {lb: {t: segment_result(t, tickers[t], lb, cost_cfg("realistic"))
                 for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: per_cost["realistic"][str(lb)][t] for t in target}
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
        cost_levels={k: dict(v) for k, v in COST_LEVELS.items()},
        per_cost=per_cost,
        summary=summary,
        cost_class_counts=class_counts,
        cost_model_control=control,
        cost_model_confound=bool(n_ctrl_up > 0),
        cost_model_confound_detail={
            "constant_long_return_rises_with_cost": int(n_ctrl_up),
            "n_tickers": len(ticker_order),
            "reason": (
                "engine prices every fill at close + slippage for both buys and "
                "sells, so a net-selling (long-biased) strategy is credited the "
                "slippage on its trims; a genuine cost cannot raise return"
            ),
        },
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())