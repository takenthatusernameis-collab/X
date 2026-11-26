"""Real-data transaction-cost sensitivity of the short-horizon momentum class.

Frontier cell R-001 (campaign slot 02). The momentum class was admitted as
candidate positive evidence (activations 37318950814 / 37361000967) with an
explicit qualification: the edge is short-horizon and mechanical, and its
execution-cost robustness had not been tested. This check closes that
qualification with a bounded cost-sensitivity test on the collected 10-asset
universe.

Primary question: does the short-horizon momentum edge survive conservative
transaction-cost stress on the collected real-data universe?

Design (fixed a-priori, not tuned to any cost result):
- Momentum = long the previous ``lookback``-day return, hold 1 day, daily
  rebalance (``bt.momentum_signals``), lookbacks {3, 5, 10}.
- Same walk-forward and 4-block volatility regime family as the admitted
  momentum artifacts (train=252d/test=84d/warmup=60d/overlap=60d, min 400
  bars/segment), each lookback vs the framework's coin-flip null on the same
  segments (``bt.stress_segments`` with a cost ``cfg`` threaded through both
  the candidate and the null).
- Bounded cost grid (the repository's existing realistic cost model, from
  ``examples/ma_crossover_real_data.py``, plus a 2x conservative stress):
    zero        : no commission / slippage (reference)
    realistic   : commission_per_trade=2.0, commission_per_share=0.003,
                  slippage_cents=2.0, slippage_proportional=0.0005  (1x repo model)
    conservative: 2x every cost parameter (stress)
- Cost-robustness is judged from the candidate's OWN absolute per-segment
  median log returns (does the positive REGIME_STABLE edge survive in
  absolute terms), NOT from the candidate-minus-null gap. This is deliberate:
  the framework's coin-flip null (``random_signals``) has far higher turnover
  than a stable-sign momentum bet, so under costs the null is penalised much
  more than the candidate and the gap widens. The candidate's cost
  robustness is therefore established from its own medians; the null
  asymmetry is recorded as a documented caveat, not used as evidence.

A-priori prediction (recorded before execution): the momentum bet is low
turnover (stable sign -> few sign flips; only the mechanical constant-dollar
rebalance trades), so the candidate's per-fold median log return will fall by
only a small amount (order 0.001-0.01) under the 1x and 2x cost levels, and
the positive REGIME_STABLE edge is predicted COST_ROBUST for >= 4 of the 6
zero-cost-robust assets. Falsification: if fewer than 4 of the 6 zero-cost
robust assets remain COST_ROBUST under conservative costs (candidate medians
collapse to <= 0 or the verdict degrades), the cost-robustness claim is
REJECTED.

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

# Repository's existing realistic cost model (examples/ma_crossover_real_data.py)
# plus a 2x conservative stress and a zero-cost reference.
COST_CONFIGS = {
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010),
}
COST_ORDER = ("zero", "realistic", "conservative")


def make_cfg(cost_name: str) -> bt.BacktestConfig:
    c = COST_CONFIGS[cost_name]
    return bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARM,
        commission_per_trade=c["commission_per_trade"],
        commission_per_share=c["commission_per_share"],
        slippage_cents=c["slippage_cents"],
        slippage_proportional=c["slippage_proportional"],
    )


def segment_result(ticker, bars, lookback, cost_name):
    """Walk-forward regime-stability result for one ticker/lookback/cost level."""
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
        cfg=make_cfg(cost_name),
    )
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 6) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 6) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 6),
        null_dispersion=round(res.null_dispersion, 6),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def turnover_stats(ticker, bars, lookback, cost_name):
    """Full-sample fills and dollar turnover for the candidate at a cost level."""
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=lookback)
    res = bt.run_bars(list(bars), sig, make_cfg(cost_name))
    turnover = float(sum(abs(f.shares) * f.price for f in res.trades))
    return dict(fills=len(res.trades), turnover_dollars=round(turnover, 2))


def lookback_verdict(sr):
    """EDGE / LOSS / NO_EDGE for one lookback's scenario result."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def edge_counts(per_lookback_cost):
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(per_lookback_cost[str(lb)])] += 1
    return counts


def cost_robustness(zero_counts, stress_counts):
    """Per-asset cost-robustness verdict from zero-cost vs stress edge counts.

    NO_EDGE_ZERO_COST : the asset was not lookback-robust at zero cost
                        (fewer than 2 EDGE lookbacks) -> cost test not
                        decision-relevant for admission.
    COST_ROBUST       : >= 2 EDGE lookbacks at zero cost AND >= 2 at stress.
    COST_DEGRADED     : >= 2 EDGE at zero cost but exactly 1 at stress.
    COST_REJECTED     : >= 2 EDGE at zero cost but 0 at stress.
    """
    if zero_counts["EDGE"] < 2:
        return "NO_EDGE_ZERO_COST"
    if stress_counts["EDGE"] >= 2:
        return "COST_ROBUST"
    if stress_counts["EDGE"] == 1:
        return "COST_DEGRADED"
    return "COST_REJECTED"


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

    print("\n=== 3. Leakage review (lookback=5, realistic cost) ===")
    for ticker in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), sig, make_cfg("realistic"))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, signals {len(sig)}, leakage [PASS], "
              f"realistic-cost total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only.")

    print("\n=== 4. Cost-sensitivity grid (lookback 3/5/10 x zero/realistic/conservative) ===")
    print("  cost model (repo realistic, from examples/ma_crossover_real_data.py):")
    for name in COST_ORDER:
        c = COST_CONFIGS[name]
        print(f"    {name:12s} commission/trade={c['commission_per_trade']} "
              f"commission/share={c['commission_per_share']} "
              f"slippage_cents={c['slippage_cents']} "
              f"slippage_prop={c['slippage_proportional']}")
    print("  walk-forward per segment train={}d/test={}d/warmup={}d/overlap={}d, "
          "min {} bars/segment, 4 volatility blocks, matched coin-flip null\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(all_tickers.keys())

    per_asset = {}
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        entry = {"turnover": {}, "lookback": {}}
        for lb in LOOKBACKS:
            entry["lookback"][str(lb)] = {}
            for cost_name in COST_ORDER:
                entry["lookback"][str(lb)][cost_name] = segment_result(
                    ticker, bars, lb, cost_name)
        for cost_name in COST_ORDER:
            entry["turnover"][cost_name] = turnover_stats(ticker, bars, 5, cost_name)
        # Classify cost robustness (realistic and conservative) vs zero-cost base.
        def counts_for(cost_name):
            return edge_counts({str(lb): entry["lookback"][str(lb)][cost_name]
                                for lb in LOOKBACKS})
        cz = counts_for("zero")
        cr = counts_for("realistic")
        cc = counts_for("conservative")
        entry["edge_counts"] = {"zero": cz, "realistic": cr, "conservative": cc}
        entry["zero_cost_robust"] = cz["EDGE"] >= 2
        entry["cost_verdict_realistic"] = cost_robustness(cz, cr)
        entry["cost_verdict_conservative"] = cost_robustness(cz, cc)
        per_asset[ticker] = entry

    # Universe-level classification.
    robust_assets = [t for t in ticker_order if per_asset[t]["zero_cost_robust"]]
    cons_counts = {"COST_ROBUST": 0, "COST_DEGRADED": 0, "COST_REJECTED": 0}
    real_counts = {"COST_ROBUST": 0, "COST_DEGRADED": 0, "COST_REJECTED": 0}
    for t in robust_assets:
        cons_counts[per_asset[t]["cost_verdict_conservative"]] += 1
        real_counts[per_asset[t]["cost_verdict_realistic"]] += 1

    def universe_class(counts):
        if counts["COST_ROBUST"] >= 4:
            return "COST_ROBUST"
        if counts["COST_ROBUST"] >= 1:
            return "COST_DEGRADED"
        return "COST_REJECTED"

    cons_class = universe_class(cons_counts)
    real_class = universe_class(real_counts)

    # Report.
    print("  Per-asset candidate medians (zero -> realistic -> conservative) and verdicts:")
    for ticker in ticker_order:
        e = per_asset[ticker]
        print(f"    {ticker}: zero_robust={e['zero_cost_robust']} "
              f"cons_verdict={e['cost_verdict_conservative']}")
        for lb in LOOKBACKS:
            cols = []
            for cost_name in COST_ORDER:
                sr = e["lookback"][str(lb)][cost_name]
                cols.append("{}:{}".format(cost_name, sr["medians"]))
            print("      lb={} {}".format(lb, "  ".join(cols)))
        tz = e["turnover"]["zero"]["turnover_dollars"]
        tc = e["turnover"]["conservative"]["turnover_dollars"]
        print(f"      turnover$ zero={tz:.0f} conservative={tc:.0f} "
              f"fills={e['turnover']['zero']['fills']}")
    print("")
    print(f"  zero-cost lookback-robust assets ({len(robust_assets)}): {robust_assets}")
    print(f"  conservative-cost outcome counts: {cons_counts} -> {cons_class}")
    print(f"  realistic-cost outcome counts:    {real_counts} -> {real_class}")

    # Determinism: rerun AMZN subset and assert identical.
    res2 = {lb: {c: segment_result("AMZN", all_tickers["AMZN"], lb, c) for c in COST_ORDER}
            for lb in LOOKBACKS}
    out1 = json.dumps({str(lb): per_asset["AMZN"]["lookback"][str(lb)] for lb in LOOKBACKS},
                      sort_keys=True)
    out2 = json.dumps({str(lb): res2[lb] for lb in LOOKBACKS}, sort_keys=True)
    print("\ndeterminism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_configs={k: COST_CONFIGS[k] for k in COST_ORDER},
        per_asset=per_asset,
        universe=dict(
            ticker_order=ticker_order,
            zero_cost_robust_assets=robust_assets,
            conservative_counts=cons_counts,
            realistic_counts=real_counts,
            classification_conservative=cons_class,
            classification_realistic=real_class,
        ),
        a_priori_prediction=(
            "COST_ROBUST for >=4 of the zero-cost-robust assets; small absolute "
            "drag (order 0.001-0.01); null gap widens under costs (null turnover "
            "asymmetry)."
        ),
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