"""Cost-sensitivity check for short-horizon momentum on the collected universe.

Objective (campaign task R-001): run a bounded cost-sensitivity check for
lookback 3/5/10 momentum under the repository's existing realistic cost model
and a matched coin-flip null.

Background: the momentum frontier cell (`research/checks/momentum.py`,
activation 37318950814) was admitted as candidate positive evidence with a
REGIME_STABLE positive edge in 7/10 of the collected universe, and the lookback
sweep (`research/checks/momentum_lookback_sweep.py`, activation 37361000967)
found the edge is not a single-point lookback artifact. Both runs used the
engine's DEFAULT (zero-cost) BacktestConfig. The remaining decision-relevant
qualification is whether that positive edge survives realistic transaction
costs, because momentum is a mechanical, daily-rebalanced 1-day bet whose
turnover could be eliminated by commissions and slippage.

Design (predeclared, fixed a-priori, NOT tuned to outcomes):

Cost grid. A single multiplicative scale applied to the repository's existing
realistic cost vector, taken verbatim from `examples/ma_crossover_real_data.py`:

    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005

Scales evaluated for every (ticker, lookback):
    scale 0.0 -> all cost parameters zero (reference; must reproduce the
                 existing zero-cost momentum_lookback_sweep artifact exactly).
    scale 1.0 -> the repository's realistic cost model.
    scale 2.0 -> conservative stress (double the realistic model).

Lookbacks: 3, 5, 10 (identical to the lookback sweep). Same walk-forward
windows (train=252d/test=84d/warmup=60d/overlap=60d), same 4 contiguous
past-only volatility blocks, same min 400 bars/segment, same matched coin-flip
null (run through the same cost cfg, so the null is cost-matched).

Predeclared cost-survival classification (per lookback, comparing scale 1.0 to
the scale 0.0 baseline), using:
    pos_edge_count = number of assets whose segment verdict is REGIME_STABLE and
                     whose candidate medians are all > 0 (a positive stable edge);
    mean_cand_median = mean of the per-segment candidate medians over all assets.

    SURVIVES  : scale1 pos_edge_count >= scale0 pos_edge_count - 1 AND
                scale1 mean_cand_median > 0.
    DEGRADES  : scale1 pos_edge_count <  scale0 pos_edge_count - 1 AND
                scale1 mean_cand_median > 0.
    COLLAPSES : scale1 mean_cand_median <= 0 OR scale1 pos_edge_count == 0.

This rule is declared here, before execution; the check reports it but the
authoritative output is the numeric cost curve, which the independent verifier
recomputes. Research / simulation only. No live trading or production execution.
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
COST_SCALES = (0.0, 1.0, 2.0)
BASE_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
ZERO_COST_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def cost_config(scale):
    """BacktestConfig with every cost parameter scaled by `scale`.

    scale 0.0 reproduces the engine's default zero-cost config exactly.
    """
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=BASE_COST["commission_per_trade"] * scale,
        commission_per_share=BASE_COST["commission_per_share"] * scale,
        slippage_cents=BASE_COST["slippage_cents"] * scale,
        slippage_proportional=BASE_COST["slippage_proportional"] * scale,
    )


def segment_result(ticker, bars, lookback, scale):
    """Walk-forward regime-stability result for one ticker, lookback, cost scale.

    Mirrors `momentum_lookback_sweep.segment_result` but applies a cost-scaled
    BacktestConfig to both the candidate and the matched coin-flip null.
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
        cfg=cost_config(scale),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
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


def is_positive_edge(sr):
    """True when the segment result is a REGIME_STABLE positive edge."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def aggregate(ticker_order, per_lb_scale):
    """Aggregate pos-edge count and mean medians for one (lookback, scale)."""
    pos_edge = 0
    pos_seg = 0
    total_seg = 0
    all_med = []
    all_null = []
    for t in ticker_order:
        sr = per_lb_scale[t]
        if is_positive_edge(sr):
            pos_edge += 1
        for m in sr["medians"]:
            all_med.append(m)
            if m > 0:
                pos_seg += 1
            total_seg += 1
        for m in sr["null_medians"]:
            all_null.append(m)
    return dict(
        pos_edge_count=pos_edge,
        pos_segment_count=pos_seg,
        total_segment_count=total_seg,
        pos_segment_fraction=round(pos_seg / total_seg, 3) if total_seg else None,
        mean_candidate_median=round(float(np.mean(all_med)), 4) if all_med else None,
        mean_null_median=round(float(np.mean(all_null)), 4) if all_null else None,
    )


def cost_survival(agg0, agg1):
    """Predeclared classification of cost survival at scale 1.0 vs scale 0.0."""
    if agg1["mean_candidate_median"] is None or agg1["pos_edge_count"] == 0 \
            or agg1["mean_candidate_median"] <= 0:
        return "COLLAPSES"
    if agg1["pos_edge_count"] >= agg0["pos_edge_count"] - 1:
        return "SURVIVES"
    return "DEGRADES"


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

    print("\n=== 4. Cost-sensitivity sweep (lookbacks 3/5/10 x cost scales 0/1/2) ===")
    print("  Cost grid (scale applied to realistic base commission_per_trade=2.0, "
          "commission_per_share=0.003, slippage_cents=2.0, "
          "slippage_proportional=0.0005):")
    print("    scale 0.0 = zero cost (reference)")
    print("    scale 1.0 = repository realistic model")
    print("    scale 2.0 = conservative stress (2x realistic)")
    print("  Walk-forward train={}d / test={}d / warmup={}d / overlap={}d, "
          "min {} bars/segment, 4 contiguous volatility blocks, "
          "each lookback x scale vs a cost-matched coin-flip null\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][scale][ticker].
    results = {
        lb: {scale: {t: segment_result(t, all_tickers[t], lb, scale)
                     for t in ticker_order}
             for scale in COST_SCALES}
        for lb in LOOKBACKS
    }

    # Aggregate + predeclared cost-survival classification.
    print("  Cost curve by lookback (pos_edge_count = REGIME_STABLE & all "
          "medians > 0; mean_cand = mean per-segment candidate median):")
    summary = {}
    for lb in LOOKBACKS:
        agg_by_scale = {scale: aggregate(ticker_order, results[lb][scale])
                        for scale in COST_SCALES}
        survival = cost_survival(agg_by_scale[0.0], agg_by_scale[1.0])
        summary[str(lb)] = dict(
            by_scale={str(scale): agg_by_scale[scale] for scale in COST_SCALES},
            cost_survival=survival,
        )
        print(f"    lookback {lb}:")
        for scale in COST_SCALES:
            a = agg_by_scale[scale]
            print("      scale {:.1f}: pos_edge_count={:2d} pos_seg_frac={:.3f} "
                  "mean_cand={:+.4f} mean_null={:+.4f}".format(
                      scale, a["pos_edge_count"], a["pos_segment_fraction"],
                      a["mean_candidate_median"], a["mean_null_median"]))
        print(f"      -> cost_survival (scale 1.0 vs 0.0): {survival}")
    print("")

    # Per-asset detail table at scale 1.0 (realistic) for the reference lookback.
    print("  Per-asset detail at scale 1.0 (realistic costs):")
    for lb in LOOKBACKS:
        print(f"    lookback {lb}:")
        for t in ticker_order:
            sr = results[lb][1.0][t]
            med = ", ".join(f"{m:+.3f}" for m in sr["medians"])
            print("      {:6s} segments={:<22} medians=[{}] verdict={}".format(
                t, ",".join(sr["segments"]), med, sr["verdict"]))
    print("")

    # Determinism: rerun the AMZN/JPM subset for every lookback at scale 1.0 and
    # scale 2.0 and assert identical output to the first pass.
    res2 = {lb: {scale: {t: segment_result(t, tickers[t], lb, scale) for t in target}
                 for scale in (1.0, 2.0)}
            for lb in LOOKBACKS}
    out1 = json.dumps(
        {lb: {str(scale): {t: results[lb][scale][t] for t in target}
              for scale in (1.0, 2.0)}
         for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(
        {lb: {str(scale): {t: res2[lb][scale][t] for t in target}
              for scale in (1.0, 2.0)}
         for lb in LOOKBACKS}, sort_keys=True)
    print("determinism (AMZN/JPM, lookbacks 3/5/10, scales 1.0/2.0): r1 == r2: {}".format(
        out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Zero-cost cross-consistency gate: scale 0.0 must reproduce the existing
    # zero-cost lookback-sweep artifact exactly (medians, nulls, verdicts).
    print("\n=== 5. Zero-cost cross-consistency vs momentum_lookback_sweep_results.json ===")
    zero_ok = True
    if ZERO_COST_ARTIFACT.exists():
        with open(ZERO_COST_ARTIFACT) as f:
            zero = json.load(f)
        for t in ticker_order:
            for lb in LOOKBACKS:
                pub = zero["per_asset"][t][str(lb)]
                mine = results[lb][0.0][t]
                match = (mine["medians"] == [round(v, 3) for v in pub["medians"]]
                         and mine["null_medians"] == [round(v, 3) for v in pub["null_medians"]]
                         and mine["verdict"] == pub["verdict"])
                if not match:
                    zero_ok = False
                    print("  MISMATCH {} lookback {}: mine medians {} nulls {} verdict {} "
                          "vs pub medians {} nulls {} verdict {}".format(
                              t, lb, mine["medians"], mine["null_medians"],
                              mine["verdict"], pub["medians"], pub["null_medians"],
                              pub["verdict"]))
        print("  zero-cost scale 0.0 reproduces the existing artifact: {}".format(zero_ok))
    else:
        print("  existing zero-cost artifact not found; cross-consistency gate skipped")
        zero_ok = None

    # Write the artifact.
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_scales=list(COST_SCALES),
        base_cost=BASE_COST,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        summary=summary,
        zero_cost_cross_consistency=zero_ok,
        per_asset={
            str(lb): {
                str(scale): {
                    t: dict(
                        segments=results[lb][scale][t]["segments"],
                        medians=results[lb][scale][t]["medians"],
                        null_medians=results[lb][scale][t]["null_medians"],
                        candidate_dispersion=results[lb][scale][t]["candidate_dispersion"],
                        null_dispersion=results[lb][scale][t]["null_dispersion"],
                        n_folds=results[lb][scale][t]["n_folds"],
                        verdict=results[lb][scale][t]["verdict"],
                    )
                    for t in ticker_order
                }
                for scale in COST_SCALES
            }
            for lb in LOOKBACKS
        },
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