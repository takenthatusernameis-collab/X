"""Real-data transaction-cost sensitivity for the momentum frontier cell.

Objective (fixed a-priori): classify whether the candidate positive momentum
edge (long the previous lookback-day return, hold 1 day, daily rebalance) that
was admitted as candidate positive evidence at lookbacks 3/5/10 survives the
repository's *existing* realistic cost model, under a cost-matched coin-flip
null. No signal rule is changed, the universe is not expanded, and no leverage
is introduced.

Cost model (the repository's existing "realistic costs" config, identical to
``examples/ma_crossover_real_data.py`` and ``examples/ma_crossover.py``):
    commission_per_trade = 2.0
    commission_per_share = 0.003
    slippage_cents       = 2.0
    slippage_proportional = 0.0005   (5 bps)
The zero-cost reference uses the default ``BacktestConfig()`` (all costs 0).

For every collected ticker and lookback in {3, 5, 10} the momentum signal is
walk-forward validated inside the same 4 contiguous volatility blocks as
``momentum_lookback_sweep.py`` (train=252d/test=84d/warmup=60d/overlap=60d,
min 400 bars/segment), twice: once at zero cost and once at the realistic cost
model. ``stress_segments`` forwards the ``cfg`` to BOTH the candidate sweep and
the coin-flip ``noise_benchmark``, so the null is cost-matched: it pays the same
per-turnover costs on the same folds. The verdict logic is unchanged from the
momentum class (CONSISTENT_WITH_NOISE / REGIME_DEPENDENT / REGIME_STABLE_LOSS /
REGIME_STABLE).

Cost-robustness classification (fixed a-priori, not tuned to any result):
    cell_edge(result) = EDGE  if result.verdict == REGIME_STABLE and all medians > 0
                        LOSS  if result.verdict == REGIME_STABLE_LOSS
                        NO_EDGE otherwise
    cell_class =
        NO_EDGE_BASELINE  if the ZERO-cost cell is not EDGE (nothing to stress)
        COST_ROBUST       if the zero-cost cell is EDGE and the realistic-cost
                          cell is still EDGE
        COST_FRAGILE      if the zero-cost cell is EDGE but the realistic-cost
                          cell is no longer EDGE
``cost_drag`` per segment = zero_cost median - realistic_cost median (>= 0 means
the cost model reduces the median log return, as expected).

A-priori falsification prediction (recorded BEFORE execution): the momentum rule
is daily-rebalanced on a constant-notional target, so it turns over on almost
every bar and pays the realistic cost model's fixed + per-share commission and
proportional/fixed slippage nearly daily. I therefore predict that a MAJORITY of
the zero-cost EDGE cells will become COST_FRAGILE under the realistic cost
model, that the cost drag is roughly common across lookbacks (the lookback
changes the signal sign path, not the daily-turnover frequency), and that any
surviving COST_ROBUST cell is concentrated among assets with the largest gross
edge. This prediction is falsifiable: if most zero-cost edges survive, the cost
robustness of the momentum evidence is stronger than the a-priori expectation.

Determinism is asserted by an internal independent rerun of the AMZN/JPM subset.
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

# The repository's existing realistic cost model (see module docstring).
REALISTIC = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO = bt.BacktestConfig(initial_capital=1e6)

COST_DRAG_TOL = 0.005  # report a drag only if |drag| exceeds half a basis point
                       # in log-return units (used only for the summary count,
                       # not for the classification rule).


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker/lookback under `cfg`."""
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


def cell_edge(sr):
    """Classify a single segment-run result's edge status (EDGE/LOSS/NO_EDGE)."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def cell_class(zero_sr, cost_sr):
    """Cost-robustness class for one (asset, lookback) cell."""
    z_edge = cell_edge(zero_sr)
    if z_edge != "EDGE":
        return "NO_EDGE_BASELINE"
    return "COST_ROBUST" if cell_edge(cost_sr) == "EDGE" else "COST_FRAGILE"


def cost_drag(zero_medians, cost_medians):
    """Per-segment zero-cost minus realistic-cost median log return."""
    return [round(z - c, 3) for z, c in zip(zero_medians, cost_medians)]


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

    print("\n=== 4. Cost-sensitivity sweep (zero vs realistic cost, lookback 3/5/10) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks; each cell judged vs a coin-flip null "
          "at the SAME cost model\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))
    print("  realistic cost model: commission_per_trade=2.0, "
          "commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005\n")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    per_asset = {}
    class_counts = {"NO_EDGE_BASELINE": 0, "COST_ROBUST": 0, "COST_FRAGILE": 0}
    drag_all = []
    zero_edge_cells = 0
    for ticker in ticker_order:
        profile = {}
        for lb in LOOKBACKS:
            zero_sr = segment_result(ticker, all_tickers[ticker], lb, ZERO)
            cost_sr = segment_result(ticker, all_tickers[ticker], lb, REALISTIC)
            cls = cell_class(zero_sr, cost_sr)
            drag = cost_drag(zero_sr["medians"], cost_sr["medians"])
            profile[str(lb)] = dict(
                zero=dict(
                    segments=zero_sr["segments"], medians=zero_sr["medians"],
                    null_medians=zero_sr["null_medians"],
                    verdict=zero_sr["verdict"],
                    candidate_dispersion=zero_sr["candidate_dispersion"],
                    null_dispersion=zero_sr["null_dispersion"],
                    n_folds=zero_sr["n_folds"],
                    edge=cell_edge(zero_sr),
                ),
                cost=dict(
                    segments=cost_sr["segments"], medians=cost_sr["medians"],
                    null_medians=cost_sr["null_medians"],
                    verdict=cost_sr["verdict"],
                    candidate_dispersion=cost_sr["candidate_dispersion"],
                    null_dispersion=cost_sr["null_dispersion"],
                    n_folds=cost_sr["n_folds"],
                    edge=cell_edge(cost_sr),
                ),
                cost_drag=drag,
                cell_class=cls,
            )
            class_counts[cls] += 1
            drag_all.extend(drag)
            if cls != "NO_EDGE_BASELINE":
                zero_edge_cells += 1
        # Per-asset cost-robustness roll-up across lookbacks.
        robust_lb = [lb for lb in LOOKBACKS
                     if profile[str(lb)]["cell_class"] == "COST_ROBUST"]
        fragile_lb = [lb for lb in LOOKBACKS
                      if profile[str(lb)]["cell_class"] == "COST_FRAGILE"]
        baseline_lb = [lb for lb in LOOKBACKS
                       if profile[str(lb)]["cell_class"] == "NO_EDGE_BASELINE"]
        if robust_lb:
            asset_class = "COST_ROBUST"
        elif fragile_lb:
            asset_class = "COST_FRAGILE"
        else:
            asset_class = "NO_EDGE"
        per_asset[ticker] = dict(
            by_lookback=profile,
            robust_lookbacks=robust_lb,
            fragile_lookbacks=fragile_lb,
            no_edge_lookbacks=baseline_lb,
            asset_class=asset_class,
        )

    total_cells = len(ticker_order) * len(LOOKBACKS)
    drag_arr = np.array(drag_all, dtype=np.float64)
    print("  Per-asset cost-robustness profile (zero-cost EDGE cells only are "
          "stress-tested):")
    for ticker in ticker_order:
        p = per_asset[ticker]
        cells = ", ".join(
            "{}:{}->{}".format(lb, p["by_lookback"][str(lb)]["zero"]["verdict"],
                               p["by_lookback"][str(lb)]["cost"]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} class={:<14s} robust={:<8} fragile={:<8} cells[{}]".format(
            ticker, p["asset_class"], str(p["robust_lookbacks"]),
            str(p["fragile_lookbacks"]), cells))
    print("")
    print("  Cost-robustness cell counts across universe: "
          "NO_EDGE_BASELINE={} | COST_ROBUST={} | COST_FRAGILE={} "
          "(of {} cells)".format(
              class_counts["NO_EDGE_BASELINE"], class_counts["COST_ROBUST"],
              class_counts["COST_FRAGILE"], total_cells))
    print("  Zero-cost EDGE cells stress-tested: {} (COST_ROBUST / COST_FRAGILE "
          "= {} / {})".format(
              zero_edge_cells,
              class_counts["COST_ROBUST"], class_counts["COST_FRAGILE"]))
    if drag_arr.size:
        print("  Cost drag (zero - realistic median log return) across all "
              "{} segment cells: median {:+.4f}, mean {:+.4f}, min {:+.4f}, "
              "max {:+.4f}".format(
                  drag_arr.size, float(np.median(drag_arr)),
                  float(np.mean(drag_arr)), float(drag_arr.min()),
                  float(drag_arr.max())))
        print("  Segment cells with |drag| > {:.3f}: {}".format(
            COST_DRAG_TOL, int(np.sum(np.abs(drag_arr) > COST_DRAG_TOL))))
    print("")

    # Determinism: rerun the target subset and assert identical output. The
    # rerun rebuilds the same per-cell profile (zero, cost, cost_drag,
    # cell_class) that is stored for the published run, so the comparison is
    # structurally identical.
    res2 = {}
    for ticker in target:
        res2[ticker] = {}
        for lb in LOOKBACKS:
            zero_sr = segment_result(ticker, tickers[ticker], lb, ZERO)
            cost_sr = segment_result(ticker, tickers[ticker], lb, REALISTIC)
            res2[ticker][str(lb)] = dict(
                zero=dict(
                    segments=zero_sr["segments"], medians=zero_sr["medians"],
                    null_medians=zero_sr["null_medians"],
                    verdict=zero_sr["verdict"],
                    candidate_dispersion=zero_sr["candidate_dispersion"],
                    null_dispersion=zero_sr["null_dispersion"],
                    n_folds=zero_sr["n_folds"],
                    edge=cell_edge(zero_sr),
                ),
                cost=dict(
                    segments=cost_sr["segments"], medians=cost_sr["medians"],
                    null_medians=cost_sr["null_medians"],
                    verdict=cost_sr["verdict"],
                    candidate_dispersion=cost_sr["candidate_dispersion"],
                    null_dispersion=cost_sr["null_dispersion"],
                    n_folds=cost_sr["n_folds"],
                    edge=cell_edge(cost_sr),
                ),
                cost_drag=cost_drag(zero_sr["medians"], cost_sr["medians"]),
                cell_class=cell_class(zero_sr, cost_sr),
            )
    out1 = json.dumps({t: {str(lb): per_asset[t]["by_lookback"][str(lb)]
                           for lb in LOOKBACKS} for t in target}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_cost=dict(
            commission_per_trade=REALISTIC.commission_per_trade,
            commission_per_share=REALISTIC.commission_per_share,
            slippage_cents=REALISTIC.slippage_cents,
            slippage_proportional=REALISTIC.slippage_proportional,
        ),
        cost_drag_tol=COST_DRAG_TOL,
        per_asset=per_asset,
        universe=dict(
            ticker_order=ticker_order,
            class_counts=class_counts,
            total_cells=total_cells,
            zero_edge_cells=zero_edge_cells,
            cost_drag_median=(round(float(np.median(drag_arr)), 4)
                              if drag_arr.size else None),
            cost_drag_mean=(round(float(np.mean(drag_arr)), 4)
                            if drag_arr.size else None),
            cost_drag_min=(round(float(drag_arr.min()), 4)
                           if drag_arr.size else None),
            cost_drag_max=(round(float(drag_arr.max()), 4)
                           if drag_arr.size else None),
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