"""Real-data transaction-cost sensitivity of the short-horizon momentum edge.

Objective (focused task R-001): the lookback 3/5/10 momentum edge was admitted
as candidate positive evidence (activations 37318950814 / 37361000967) but was
validated at ZERO transaction cost. The remaining decision-relevant
qualification is whether that edge survives conservative execution costs on the
collected 10-asset universe. This check runs a bounded, predeclared cost grid
through the SAME walk-forward regime-stability gate and matched coin-flip null
used by `research/checks/momentum_lookback_sweep.py`, and classifies the
resulting cost robustness.

Predeclared cost grid (a-priori, NOT tuned to results):
  - zero        : BacktestConfig() (no costs). Reference anchor; the lookback
                  3/5/10 medians/nulls/segments MUST exactly reproduce the
                  prior `momentum_lookback_sweep_results.json` artifact.
  - realistic   : the repository's canonical realistic cost model, identical to
                  the one used in `examples/ma_crossover.py`,
                  `examples/volatility_regime_filter.py` and
                  `examples/ma_crossover_real_data.py`:
                  commission_per_trade=2.0, commission_per_share=0.003,
                  slippage_cents=2.0, slippage_proportional=0.0005 (5 bps).
  - conservative: 2x the realistic model (pessimistic stress):
                  commission_per_trade=4.0, commission_per_share=0.006,
                  slippage_cents=4.0, slippage_proportional=0.001 (10 bps).

The matched null is the framework's coin-flip null (`noise_benchmark`) run on
the same segments with the SAME cost cfg, so both the candidate and the null pay
identical costs. Costs are charged by the existing engine on every rebalance
fill (the constant-dollar sizing rebalances daily), so this stress tests the
implemented strategy's turnover honestly; it does not alter the signal or sizing.

Predeclared classification (a-priori, applied at the canonical lookback 5 and
reported at 3/5/10):
  EDGE(segment result) := verdict == "REGIME_STABLE" and all(segment medians>0)
  E0 = #assets EDGE at zero cost; ER = #assets EDGE at realistic cost;
  EC = #assets EDGE at conservative cost.
  - COST_ROBUST    : ER >= 0.6 * E0   (>=60% of the zero-cost edge assets survive
                                        the realistic cost model)
  - COST_FRAGILE   : 0 < ER < 0.6 * E0
  - COST_ELIMINATED: ER == 0
  - NO_ZERO_COST_EDGE: E0 == 0 (nothing to stress)

A-priori prediction (recorded BEFORE running, not adjusted afterwards): because
the implemented strategy rebalances to constant-dollar exposure every bar, its
turnover is high and the realistic model will charge costs on essentially every
bar; the ~0.05-0.13 per-fold log-return edge is of the same order as realistic
drag, so the edge is expected to be MATERIALLY REDUCED at the realistic level
(COST_FRAGILE or COST_ELIMINATED). A COST_ROBUST outcome would strengthen the
momentum admission; COST_ELIMINATED would qualify it as cost-dependent. The
zero-cost column reproducing the prior artifact anchors the whole pipeline.

Method: long the previous lookback-day return; hold 1 day; daily rebalance;
lookbacks 3/5/10; walk-forward per segment train=252d/test=84d/warmup=60d/
overlap=60d; 4 contiguous volatility blocks; min 400 bars/segment. Determinism is
asserted via an internal independent recomputation of the AMZN/JPM subset.

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
CANONICAL_LOOKBACK = 5
SURVIVAL_THRESHOLD = 0.6  # a-priori: >=60% of zero-cost edges must survive realistic costs

# Predeclared cost grid. `zero` is the default (costless) BacktestConfig anchor.
COST_GRID = {
    "zero": {},
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
        slippage_proportional=0.001,
    ),
}
COST_LEVELS = ("zero", "realistic", "conservative")

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
PRIOR_LOOKBACK_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def cost_config(level):
    """BacktestConfig for a predeclared cost level."""
    return bt.BacktestConfig(**COST_GRID[level])


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker/lookback/cost level.

    Uses the framework `stress_segments` (identical pipeline to
    `momentum_lookback_sweep.py`), differing only by the injected cost cfg.
    """
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
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def is_edge(sr):
    """A positive, regime-stable edge: REGIME_STABLE with all segment medians > 0."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def classify(E0, ER):
    """Predeclared cost-robustness classification from zero- and realistic-edge counts."""
    if E0 == 0:
        return "NO_ZERO_COST_EDGE"
    if ER >= SURVIVAL_THRESHOLD * E0:
        return "COST_ROBUST"
    if ER > 0:
        return "COST_FRAGILE"
    return "COST_ELIMINATED"


def cross_check_zero_cost(results, prior):
    """Compare the zero-cost column against the prior lookback-sweep artifact.

    Returns (status, details). A mismatch means the pipeline drifted from the
    validated zero-cost result and the cost columns cannot be trusted.
    """
    details = []
    ok = True
    for lb in LOOKBACKS:
        for ticker, prior_lb in prior["per_asset"].items():
            pub = prior_lb[str(lb)]
            mine = results["zero"][lb][ticker]
            for key in ("medians", "null_medians", "segments"):
                if mine[key] != pub[key]:
                    ok = False
                    details.append(
                        "{}/lb{}/{}: mine={} pub={}".format(ticker, lb, key, mine[key], pub[key]))
    return ("MATCH" if ok else "MISMATCH"), details


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

    print("\n=== 3. Leakage review + cost-accounting audit (AMZN, JPM) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        for lb in LOOKBACKS:
            signals = bt.momentum_signals(closes, lookback=lb)
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        # Cost-accounting audit: equity must be fully explained by fills even
        # with realistic costs (validates commission + slippage accounting).
        sig5 = bt.momentum_signals(closes, lookback=CANONICAL_LOOKBACK)
        res_real = bt.run_bars(list(bars), sig5, cost_config("realistic"))
        bt.check_equity_matches_fills(res_real.equity_curve, res_real.trades, closes, 1e6)
        total_return = (res_real.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals (3/5/10) {len(sig5)}, "
              f"leakage [PASS], equity-fill audit realistic [PASS], "
              f"net_total_return(realistic) {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Cost-sensitivity grid (lookbacks 3/5/10 x cost levels) ===")
    print("  Cost grid: " + ", ".join(
        "{}={}".format(lv, COST_GRID[lv] if COST_GRID[lv] else "zero") for lv in COST_LEVELS))
    print("  Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 contiguous volatility blocks, "
          "candidate and matched coin-flip null both pay the same cost cfg\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # results[cost_level][lookback][ticker] = segment_result dict
    results = {lv: {lb: {} for lb in LOOKBACKS} for lv in COST_LEVELS}
    for lv in COST_LEVELS:
        cfg = cost_config(lv)
        for lb in LOOKBACKS:
            for t in ticker_order:
                results[lv][lb][t] = segment_result(t, all_tickers[t], lb, cfg)
        # Compact per-level summary
        edge_by_lb = {lb: sum(is_edge(results[lv][lb][t]) for t in ticker_order)
                      for lb in LOOKBACKS}
        print("  cost={:12s} EDGE counts by lookback: {}".format(
            lv, {lb: edge_by_lb[lb] for lb in LOOKBACKS}))

    # === Zero-cost cross-consistency gate ===
    print("\n=== 5. Zero-cost cross-consistency vs prior lookback sweep ===")
    cross_status, cross_details = cross_check_zero_cost(results, json.load(open(PRIOR_LOOKBACK_ARTIFACT)))
    print("  zero-cost column vs momentum_lookback_sweep_results.json: {}".format(cross_status))
    for d in cross_details:
        print("    " + d)
    if cross_status != "MATCH":
        print("ZERO-COST CROSS-CHECK FAILED; cost columns are not trustworthy. Aborting.")
        sys.exit(2)

    # === Classification ===
    print("\n=== 6. Cost-robustness classification ===")
    per_lookback = {}
    for lb in LOOKBACKS:
        E0 = sum(is_edge(results["zero"][lb][t]) for t in ticker_order)
        ER = sum(is_edge(results["realistic"][lb][t]) for t in ticker_order)
        EC = sum(is_edge(results["conservative"][lb][t]) for t in ticker_order)
        zero_edge = [t for t in ticker_order if is_edge(results["zero"][lb][t])]
        real_edge = [t for t in ticker_order if is_edge(results["realistic"][lb][t])]
        cons_edge = [t for t in ticker_order if is_edge(results["conservative"][lb][t])]
        per_lookback[lb] = dict(
            E_zero=E0, E_realistic=ER, E_conservative=EC,
            survival_realistic=round(ER / E0, 3) if E0 else None,
            verdict=classify(E0, ER),
            zero_edge_assets=zero_edge,
            realistic_edge_assets=real_edge,
            conservative_edge_assets=cons_edge,
            lost_at_realistic=[t for t in zero_edge if t not in real_edge],
        )
        pb = per_lookback[lb]
        print("  lookback {}: E0={} ER={} EC={} survival={} -> {} | lost={}".format(
            lb, pb["E_zero"], pb["E_realistic"], pb["E_conservative"],
            pb["survival_realistic"], pb["verdict"], pb["lost_at_realistic"]))

    canonical = per_lookback[CANONICAL_LOOKBACK]
    primary_verdict = canonical["verdict"]

    # === Determinism: rerun AMZN/JPM subset across all levels/lookbacks ===
    res2 = {lv: {lb: {t: segment_result(t, tickers[t], lb, cost_config(lv)) for t in target}
                for lb in LOOKBACKS} for lv in COST_LEVELS}
    out1 = json.dumps({lv: {str(lb): {t: results[lv][lb][t] for t in target}
                            for lb in LOOKBACKS} for lv in COST_LEVELS}, sort_keys=True)
    out2 = json.dumps({lv: {str(lb): res2[lv][lb] for lb in LOOKBACKS}
                       for lv in COST_LEVELS}, sort_keys=True)
    determinism_ok = (out1 == out2)
    print("\n=== 7. Determinism (AMZN/JPM, all cost levels, all lookbacks) ===")
    print("  r1 == r2: {}".format(determinism_ok))
    assert determinism_ok, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        canonical_lookback=CANONICAL_LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_grid={lv: COST_GRID[lv] for lv in COST_LEVELS},
        survival_threshold=SURVIVAL_THRESHOLD,
        zero_cost_cross_check=cross_status,
        determinism=determinism_ok,
        primary_verdict=primary_verdict,
        per_lookback={str(lb): per_lookback[lb] for lb in LOOKBACKS},
        results={lv: {str(lb): results[lv][lb] for lb in LOOKBACKS} for lv in COST_LEVELS},
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("PRIMARY COST-ROBUSTNESS VERDICT (lookback {}): {}".format(
        CANONICAL_LOOKBACK, primary_verdict))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())