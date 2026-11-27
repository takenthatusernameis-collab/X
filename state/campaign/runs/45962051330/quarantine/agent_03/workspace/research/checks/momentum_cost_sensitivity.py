"""Real-data transaction-cost sensitivity of the qualified momentum edge.

Background: short-horizon momentum (long the previous N-day return, hold 1 day,
daily rebalance) was admitted as candidate positive evidence on the collected
10-asset universe (activations 37318950814 / 37361000967): 6/10 assets show a
lookback-robust positive REGIME_STABLE edge. That evidence was produced with the
framework's default ZERO-cost `BacktestConfig`. The open decision-relevant
qualification is whether the edge survives realistic execution costs, i.e.
whether it is a tradable net-of-cost effect or a gross-only artifact.

This check runs a bounded, predeclared cost grid on the SAME signal, universe,
walk-forward windows, volatility-block regime family, and matched coin-flip null
used by `momentum_lookback_sweep.py`, so the zero-cost column is directly
comparable to the admitted evidence:

  lookbacks  : {3, 5, 10}                      (as in the lookback sweep)
  cost levels: zero, realistic, stress2x
      zero      : BacktestConfig() defaults (no commission/slippage)
      realistic : the repository's existing "realistic costs" model
                  (commission_per_trade=2.0, commission_per_share=0.003,
                   slippage_cents=2.0, slippage_proportional=0.0005)
      stress2x  : 2x every realistic cost component (conservative stress bound)

For every (asset, lookback, cost level) the candidate's per-segment walk-forward
median log returns are compared against a coin-flip null run under the SAME cost
`cfg` (the framework's matched null). A per-asset cost-robustness classification
is derived from a fixed a-priori rule (below). The zero-cost column is
cross-checked against `momentum_lookback_sweep_results.json` (an independent
zero-cost artifact) as a consistency gate.

A-priori prediction (declared before the full grid run; mechanism-based):
the 1-day momentum signal on large-cap equities has LOW turnover (its sign is
persistent over the 2009-2026 sample, dominated by the long drift of the period),
so the per-bar constant-dollar rebalancing cost is small relative to the
multi-percent per-fold edge. Prediction: the edge is COST_ROBUST at realistic
costs for the assets already admitted as lookback-robust (MSFT, GOOGL, AMZN, META,
TSLA, AAPL), and the 2x stress does not erase it. Falsification condition: if
realistic costs reduce the count of positive-REGIME_STABLE lookbacks to <2 for a
majority of the previously-admitted assets (i.e. COST_ERASES_EDGE / COST_FRAGILE
dominates), the cost-robustness claim is falsified.

A-priori classification rule (fixed, not tuned to results), per asset:
  n_edge[c]  = number of lookbacks in {3,5,10} whose candidate verdict at cost
               level c is a positive REGIME_STABLE edge (all segment medians > 0).
  NO_BASELINE_EDGE : n_edge[zero] == 0  (no edge to stress; consistent with the
                     prior NO_EDGE assets JPM, JNJ, XOM, NVDA).
  COST_ROBUST      : n_edge[zero] >= 2 and n_edge[realistic] >= 2.
  COST_FRAGILE     : n_edge[zero] >= 2 and n_edge[realistic] == 1.
  COST_ERASES_EDGE : n_edge[zero] >= 2 and n_edge[realistic] == 0.
The 2x-stress column (n_edge[stress2x]) is reported as the conservative bound.

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
SWEEP_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

# Predeclared cost grid (fixed a-priori).
COST_LEVELS = [
    ("zero", bt.BacktestConfig(initial_capital=1e6)),
    ("realistic", bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005,
    )),
    ("stress2x", bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=4.0, commission_per_share=0.006,
        slippage_cents=4.0, slippage_proportional=0.001,
    )),
]
COST_NAMES = [name for name, _ in COST_LEVELS]
PRIMARY_COST = "realistic"


def cost_params(cfg: bt.BacktestConfig) -> dict:
    return dict(
        commission_per_trade=cfg.commission_per_trade,
        commission_per_share=cfg.commission_per_share,
        slippage_cents=cfg.slippage_cents,
        slippage_proportional=cfg.slippage_proportional,
    )


def segment_result(ticker, bars, lookback, cfg):
    """One (asset, lookback, cost) stress_segments result -> plain dict."""
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
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
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


def turnover_profile(bars, lookback):
    """Full-sample turnover/cost diagnostic (descriptive, not the evidence)."""
    closes = bars.closes_array()
    sig = bt.momentum_signals(closes, lookback=lookback)
    flips = sum(1 for i in range(1, len(sig)) if sig[i].weight != sig[i - 1].weight)
    cfg = bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005,
    )
    res = bt.run_bars(list(bars), sig, cfg)
    total_comm = float(sum(f.commission for f in res.trades))
    gross = float(sum(abs(f.shares) * f.price for f in res.trades))
    eq0 = float(res.equity_curve[0])
    eqT = float(res.equity_curve[-1])
    return dict(
        n_trades=len(res.trades),
        sign_flips=int(flips),
        total_commission_real=round(total_comm, 1),
        gross_notional_real=round(gross, 1),
        total_return_real=round((eqT - eq0) / eq0, 4),
    )


def lookback_verdict(sr):
    """Positive-edge / loss / no-edge label for one (asset, lookback, cost)."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def classify(n_edge):
    """A-priori per-asset cost-robustness classification (see module docstring)."""
    if n_edge["zero"] == 0:
        return "NO_BASELINE_EDGE"
    if n_edge[PRIMARY_COST] >= 2:
        return "COST_ROBUST"
    if n_edge[PRIMARY_COST] == 1:
        return "COST_FRAGILE"
    return "COST_ERASES_EDGE"


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

    print("\n=== 3. Leakage review per asset (realistic costs, lookback=5) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    real_cfg = dict(COST_LEVELS)[PRIMARY_COST]
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, real_cfg)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return(real) {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "costs applied per fill (commission + slippage).")

    print("\n=== 4. Predeclared cost grid (fixed a-priori) ===")
    for name, cfg in COST_LEVELS:
        print(f"  {name:10s}: {cost_params(cfg)}")
    print(f"  lookbacks: {list(LOOKBACKS)} | primary decision level: {PRIMARY_COST}")

    # Full grid: results[cost][lookback][ticker].
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    grid = {}
    for name, cfg in COST_LEVELS:
        grid[name] = {lb: {t: segment_result(t, all_tickers[t], lb, cfg) for t in ticker_order}
                      for lb in LOOKBACKS}
        print(f"  computed cost level '{name}' for lookbacks {list(LOOKBACKS)} x {len(ticker_order)} assets")

    print("\n=== 5. Per-asset cost-robustness classification ===")
    print("  (n_edge = # lookbacks with a positive REGIME_STABLE edge at that cost level)\n")
    per_asset = {}
    class_counts = {k: 0 for k in ("COST_ROBUST", "COST_FRAGILE", "COST_ERASES_EDGE", "NO_BASELINE_EDGE")}
    for ticker in ticker_order:
        profile = {"lookbacks": {}}
        n_edge = {name: 0 for name in COST_NAMES}
        for lb in LOOKBACKS:
            lb_block = {}
            for name in COST_NAMES:
                sr = grid[name][lb][ticker]
                lb_block[name] = sr
                if lookback_verdict(sr) == "EDGE":
                    n_edge[name] += 1
            lb_block["turnover"] = turnover_profile(all_tickers[ticker], lb)
            profile["lookbacks"][str(lb)] = lb_block
        profile["n_edge"] = n_edge
        cls = classify(n_edge)
        profile["classification"] = cls
        class_counts[cls] += 1
        # cost drag at lookback 5 (mean over segments of zero - realistic medians)
        z = grid["zero"][5][ticker]["medians"]
        r = grid[PRIMARY_COST][5][ticker]["medians"]
        drag = float(np.mean(np.array(z) - np.array(r))) if z and r else 0.0
        profile["cost_drag_l5"] = round(drag, 4)
        per_asset[ticker] = profile

        to = profile["lookbacks"]["5"]["turnover"]
        print("  {:6s} n_edge z{}/r{}/s{}  drag_l5={:+.4f}  flips={} trades={}  -> {}".format(
            ticker, n_edge["zero"], n_edge[PRIMARY_COST], n_edge["stress2x"],
            drag, to["sign_flips"], to["n_trades"], cls))

    print("\n  classification counts: {}".format(class_counts))

    # Determinism: rerun the target subset at realistic cost and compare.
    res2 = {lb: {t: segment_result(t, tickers[t], lb, real_cfg) for t in target}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: grid[PRIMARY_COST][lb][t] for t in target} for lb in LOOKBACKS},
                      sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    det_ok = out1 == out2
    print("\n=== 6. Determinism (realistic cost, AMZN/JPM, all lookbacks) ===")
    print("  r1 == r2: {}".format(det_ok))
    assert det_ok, "non-deterministic output"

    # Cross-consistency gate: zero-cost column must reproduce the lookback sweep.
    print("\n=== 7. Zero-cost cross-consistency vs momentum_lookback_sweep_results.json ===")
    xcheck_ok = True
    if SWEEP_ARTIFACT.exists():
        sweep = json.loads(SWEEP_ARTIFACT.read_text())
        sw_lp = sweep["universe"]["per_asset"]
        for ticker in ticker_order:
            for lb in LOOKBACKS:
                pub = sw_lp[ticker]["lookback_medians"][str(lb)]
                nulls = sw_lp[ticker]["lookback_null_medians"][str(lb)]
                mine = grid["zero"][lb][ticker]["medians"]
                mine_nulls = grid["zero"][lb][ticker]["null_medians"]
                ok = (mine == pub and mine_nulls == nulls)
                if not ok:
                    xcheck_ok = False
                    print("  MISMATCH {} lb{}: mine {} vs pub {}".format(ticker, lb, mine, pub))
        print("  zero-cost medians/nulls all match lookback sweep: {}".format(xcheck_ok))
    else:
        xcheck_ok = False
        print("  lookback sweep artifact missing; cross-consistency UNVERIFIED")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_levels={name: cost_params(cfg) for name, cfg in COST_LEVELS},
        primary_cost=PRIMARY_COST,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        a_priori_prediction=(
            "COST_ROBUST at realistic costs for the previously-admitted lookback-robust "
            "assets; 2x stress does not erase the edge; falsified if realistic costs cut "
            "positive-REGIME_STABLE lookbacks to <2 for a majority of admitted assets."),
        classification_rule=(
            "NO_BASELINE_EDGE if n_edge[zero]==0; COST_ROBUST if n_edge[zero]>=2 and "
            "n_edge[realistic]>=2; COST_FRAGILE if n_edge[zero]>=2 and n_edge[realistic]==1; "
            "COST_ERASES_EDGE if n_edge[zero]>=2 and n_edge[realistic]==0."),
        per_asset=per_asset,
        universe=dict(
            classification_counts=class_counts,
            ticker_order=ticker_order,
            zero_cost_cross_consistency=xcheck_ok,
        ),
        determinism=det_ok,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())