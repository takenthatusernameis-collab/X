"""Real-data transaction-cost sensitivity of the short-horizon momentum edge.

Objective (focused task R-001): the momentum class is ADMITTED AS CANDIDATE
POSITIVE EVIDENCE (lookback 5, `research/checks/momentum.py`, activation
37318950814) and lookback-robust across 3/5/10 (`momentum_lookback_sweep.py`,
activation 37361000967), but every prior momentum run used the engine's default
zero-cost ``BacktestConfig``. The decision-relevant unresolved qualification is
whether that edge survives conservative transaction-cost stress. This check
answers exactly that question and nothing else.

Cost model (the repository's existing engine cost model, `engine.BacktestConfig`):
the engine charges ``commission_per_trade + |delta_shares| * commission_per_share``
on every fill, and a symmetric ``close +/- slippage`` fill. A tooling diagnostic
run before this check established two engine mechanics that shape the design:

  1. Proportional/fixed slippage is applied symmetrically to buys and sells
     (``fill_price = close + slippage`` on both sides), so over a round trip the
     slippage component cancels and does NOT accumulate. It is therefore not a
     usable cost-stress channel for a two-sided strategy.
  2. ``commission_per_share`` is charged on every fill regardless of side, so it
     is the only accumulating, notional-proportional cost channel in the engine.

This check therefore expresses the cost grid as a **per-trade notional cost in
basis points** and implements it through the existing ``commission_per_share``
knob, calibrated per ticker as ``commission_per_share = (bps / 10000) * ref_price``
where ``ref_price`` is the ticker's median close over the sample. Because a
share's notional is ``price`` and ``ref_price`` is a fixed per-ticker constant,
the charged commission is ``~ bps`` of the traded notional. ``ref_price`` is a
cost-calibration constant only (it never enters the signal or its timing), is
fixed before the walk-forward, and is held identical across folds, per the
methodology cost-model rule. The coin-flip null is run with the SAME cfg, so the
candidate-vs-null comparison is cost-matched.

Predeclared, a-priori (NOT tuned to results):
  - Lookbacks: 3, 5, 10 (the admitted short-horizon set).
  - Cost grid (per-trade notional bps): 0, 10, 25, 50, 100.
  - Decision threshold: 25 bps/trade (~50 bps round trip), a conservative level
    for liquid large-cap US equities.
  - Falsification prediction: a mechanical, ~1x-gross, daily-rebalanced signal
    is cost-sensitive; if the edge is a fair-value/impact illusion it should
    decay toward CONSISTENT_WITH_NOISE as cost rises. Either outcome closes the
    cost-robustness qualification with a decisive verdict.

Method: for each (lookback, cost, ticker) the existing ``stress_segments`` runs
the walk-forward regime-stability gate (train=252/test=84/warmup=60/overlap=60,
4 contiguous volatility blocks, matched coin-flip null on the same segments)
under the cost cfg. Verdict and per-segment medians are recorded. Cost
robustness is classified per asset from the maximum cost level at which a
positive REGIME_STABLE edge persists.

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
COST_BPS = (0, 10, 25, 50, 100)
DECISION_BPS = 25
REFERENCE_LOOKBACK = 5
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def reference_price(closes) -> float:
    """Per-ticker cost-calibration constant: median close over the sample.

    Used only to convert a target per-trade notional cost (bps) into the
    engine's ``commission_per_share`` knob. It never enters the signal.
    """
    return float(np.median(np.asarray(closes, dtype=np.float64)))


def cost_config(bps: float, ref_price: float) -> bt.BacktestConfig:
    """Engine cfg carrying a per-trade notional cost of ``bps`` via
    ``commission_per_share`` (the only accumulating cost channel)."""
    return bt.BacktestConfig(commission_per_share=(float(bps) / 10000.0) * ref_price)


def segment_result(ticker, bars, lookback, bps, ref_price):
    """Walk-forward regime-stability result for one ticker/lookback/cost under
    the existing ``stress_segments`` gate with a cost-matched coin-flip null."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    cfg = cost_config(bps, ref_price)
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


def edge_status(sr):
    """EDGE / LOSS / NO_EDGE for one (lookback, cost) scenario result.

    A positive REGIME_STABLE edge requires the regime-stability verdict to be
    REGIME_STABLE and every segment median to be strictly positive; uniformly
    negative medians are a stable loss, not an edge.
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def cost_survival_bps(by_cost):
    """Maximum cost level (bps) at which a positive edge persists, or None.

    ``by_cost`` maps str(bps) -> scenario result for one lookback of one asset.
    """
    survivors = [b for b in COST_BPS if edge_status(by_cost[str(b)]) == "EDGE"]
    return max(survivors) if survivors else None


def classify_cost_robustness(by_lookback):
    """Per-asset cost-robustness classification at the reference lookback.

    COST_ROBUST : positive edge at 0 bps that survives to >= DECISION_BPS.
    COST_FRAGILE: positive edge at 0 bps lost before DECISION_BPS.
    NO_EDGE     : no positive edge even at 0 bps.
    """
    ref = by_lookback[str(REFERENCE_LOOKBACK)]
    e0 = edge_status(ref[str(COST_BPS[0])]) == "EDGE"
    if not e0:
        return "NO_EDGE"
    surv = cost_survival_bps(ref)
    if surv is not None and surv >= DECISION_BPS:
        return "COST_ROBUST"
    return "COST_FRAGILE"


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

    print("\n=== 3. Leakage review per asset (lookback=5, 0 bps) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=REFERENCE_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; the cost "
          "calibration constant (median close) is a fixed per-ticker scalar that "
          "never enters the signal or its timing.")

    print("\n=== 4. Cost-sensitivity sweep (lookback 3/5/10 x cost "
          f"{list(COST_BPS)} bps/trade) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 contiguous volatility blocks, "
          "cost-matched coin-flip null\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    per_asset = {}
    class_counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "NO_EDGE": 0}
    # survival-at-decision-threshold counts, per lookback, for the summary.
    survive_counts = {lb: 0 for lb in LOOKBACKS}

    for ticker in ticker_order:
        ref_price = reference_price(all_tickers[ticker].closes_array())
        by_lookback = {}
        for lb in LOOKBACKS:
            by_cost = {}
            for bps in COST_BPS:
                by_cost[str(bps)] = segment_result(
                    ticker, all_tickers[ticker], lb, bps, ref_price)
            by_lookback[str(lb)] = by_cost
        cls = classify_cost_robustness(by_lookback)
        class_counts[cls] += 1
        surv = {str(lb): cost_survival_bps(by_lookback[str(lb)]) for lb in LOOKBACKS}
        for lb in LOOKBACKS:
            s = surv[str(lb)]
            if s is not None and s >= DECISION_BPS:
                survive_counts[lb] += 1
        per_asset[ticker] = dict(
            ref_price=round(ref_price, 4),
            by_lookback=by_lookback,
            cost_survival_bps=surv,
            classification=cls,
        )
        # Compact per-asset print: lookback-5 edge status across the cost grid.
        cells = " ".join(
            "{}:{}".format(bps, edge_status(by_lookback[str(REFERENCE_LOOKBACK)][str(bps)])[0])
            for bps in COST_BPS
        )
        print(f"  {ticker:6s} ref_px={ref_price:8.2f} class={cls:12s} "
              f"surv(l5)={surv['5']} edge@lb5[{cells}]")

    print("\n  classification counts: {}".format(class_counts))
    print("  assets whose edge survives to >= {} bps/trade, by lookback: {}".format(
        DECISION_BPS, {str(lb): survive_counts[lb] for lb in LOOKBACKS}))

    # Determinism: rerun the target subset and assert identical output.
    res2 = {}
    for ticker in target:
        rp = reference_price(tickers[ticker].closes_array())
        res2[ticker] = {
            str(lb): {str(bps): segment_result(ticker, tickers[ticker], lb, bps, rp)
                      for bps in COST_BPS}
            for lb in LOOKBACKS
        }
    out1 = json.dumps({t: per_asset[t]["by_lookback"] for t in target}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("\ndeterminism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_bps=list(COST_BPS),
        decision_bps=DECISION_BPS,
        reference_lookback=REFERENCE_LOOKBACK,
        cost_model="commission_per_share = (bps/10000) * ref_price; ref_price = "
                   "median close (cost-calibration constant, fixed pre-walk-forward)",
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            classification_counts=class_counts,
            survive_to_decision_bps={str(lb): survive_counts[lb] for lb in LOOKBACKS},
            ticker_order=ticker_order,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())