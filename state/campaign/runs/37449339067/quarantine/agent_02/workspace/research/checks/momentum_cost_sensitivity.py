"""Cost-sensitivity analysis of short-horizon momentum lookbacks 3 / 5 / 10.

Primary question (R-001): Does the short-horizon momentum edge survive
conservative transaction-cost stress on the collected real-data universe?

This check is a bounded, predeclared cost-sensitivity stress test. Everything
about the grid and the cost model is fixed in advance and never tuned to the
observed results.

Contract (fixed a-priori, not tuned to OOS):
- Signal: momentum = long the previous N-day return, hold 1 day, daily
  rebalance. Same past-only construction as research/backtest/regime_stability
  (mean of log returns over the lookback window, sign applied to a weight in
  [-1, 1], neutral before the lookback window).
- Lookback grid: {3, 5, 10} (from the R-001 contract).
- Universe: the 10 collected tickers in research/data/manifest.json.
- Walk-forward per segment: train=252d / test=84d / warmup=60d / overlap=60d.
- Segments: 4 contiguous volatility blocks (past-only realized-vol
  classification), minimum 400 bars/segment.
- Cost model (predeclared, conservative realistic): the engine's
  BacktestConfig, composed of a fixed per-share slippage (cents) plus a
  proportional component (fraction of price). Three predeclared levels:
    base     : slippage_cents=0.5, slippage_proportional=0.001  (10 bps)
    stern    : slippage_cents=1.0, slippage_proportional=0.002  (20 bps)
    severe   : slippage_cents=2.0, slippage_proportional=0.005  (50 bps)
  No explicit commission_per_share: 10-50 bps proportional slippage
  conservatively absorbs round-trip execution costs for liquid large-cap
  names; no leverage beyond target_exposure=1.0.
- Matched null: the coin-flip sign null runs through the SAME
  walk-forward with the IDENTICAL cost config (inside stress_segments), so the
  pass/fail decision is cost-for-cost, never a free signal vs a costly null.
- Leakage gate: signal-integrity check and equity-fill audit on AMZN/JPM for
  the lookback=5 reference.
- Determinism: rerun the AMZN/JPM subset and assert byte-identical JSON.

Out of scope (explicitly NOT done): tuning signal rules after seeing cost
results; expanding the universe; adding leverage; changing the walk-forward
windows; changing the cost levels after the run.

Deliverable: state/check_artifacts/momentum_cost_sensitivity_results.json.
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

# Predeclared cost model: realistic conservative base, two stress levels.
# Costs are expressed as they appear in BacktestConfig: slippage_cents = fixed
# slippage per share in cents; slippage_proportional = fraction of price.
COST_LEVELS = [
    ("base", 0.5, 0.001),      # 10 bps + 0.5 cents (conservative realistic)
    ("stern", 1.0, 0.002),     # 20 bps + 1.0 cents (stress)
    ("severe", 2.0, 0.005),    # 50 bps + 2.0 cents (severe)
]

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, one lookback, one
    cost config. Returns segments / medians / null medians / n_folds / verdict
    / edge flag."""
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
        cfg=cfg,
    )
    edge = res.overall_verdict == "REGIME_STABLE" and all(
        s.baseline_median_log_return > 0 for s in res.scenarios)
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
        edge=edge,
    )


def cost_verdict(asset_profile):
    """Classify one asset's cost robustness from its base-cost lookback edges."""
    edges = [asset_profile["lookbacks"][str(lb)]["base"]["edge"] for lb in LOOKBACKS]
    n_edges = sum(edges)
    if n_edges >= 2:
        return "cost_robust"
    if n_edges == 1:
        return "cost_sensitive"
    return "cost_destroyed"


def edge_extinguished_at(asset_profile):
    """For each lookback, the FIRST cost level at which the lookback is no
    longer a REGIME_STABLE edge (base -> stern -> severe); or 'survives_severe'
    if it never dies within the grid."""
    out = {}
    levels = ["base", "stern", "severe"]
    for lb in LOOKBACKS:
        profile = asset_profile["lookbacks"]
        for lvl in levels:
            if not profile[str(lb)][lvl]["edge"]:
                out[str(lb)] = lvl
                break
        else:
            out[str(lb)] = "survives_severe"
    return out


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

    print("\n=== 3. Leakage gate (lookback=5 reference, base cost) ===")
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

    print("\n=== 4. Cost-sensitivity sweep: lookbacks 3 / 5 / 10 x "
          "3 cost levels across collected universe ===")
    print("  Cost levels: base=10bps+0.5c, stern=20bps+1.0c, severe=50bps+2.0c")
    print("  Walk-forward per segment: train=252d / test=84d / "
          "warmup=60d / overlap=60d, 4 volatility blocks, min 400 bars/seg")
    print("  Each cell = candidate momentum vs a coin-flip null WITH THE "
          "SAME COSTS (matched null)\n")

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    cost_configs = {
        name: bt.BacktestConfig(
            initial_capital=1e6,
            slippage_cents=c,
            slippage_proportional=p,
            target_exposure=1.0,
        )
        for name, c, p in COST_LEVELS
    }

    # per_asset: {ticker: {lookbacks: {lb: {lvl: cell}}, cost_robustness}}
    per_asset = {}
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        lookbacks = {}
        for lb in LOOKBACKS:
            lookbacks[str(lb)] = {
                level: segment_result(ticker, bars, lb, cost_configs[level])
                for level, _c, _p in COST_LEVELS
            }
        asset_profile = dict(lookbacks=lookbacks)
        asset_profile["cost_robustness"] = cost_verdict(asset_profile)
        asset_profile["edge_extinguished_at"] = edge_extinguished_at(asset_profile)
        per_asset[ticker] = asset_profile

    base_counts = {"cost_robust": 0, "cost_sensitive": 0, "cost_destroyed": 0}
    for ticker in ticker_order:
        base_counts[per_asset[ticker]["cost_robustness"]] += 1

    # Survival grid: for each lookback, how many assets still show an edge at
    # each cost level (used to define the class-level verdict).
    survival_grid = {}
    for lb in LOOKBACKS:
        counts = {"base": 0, "stern": 0, "severe": 0}
        for ticker in ticker_order:
            for level in ["base", "stern", "severe"]:
                counts[level] += int(per_asset[ticker]["lookbacks"][str(lb)][level]["edge"])
        survival_grid[str(lb)] = counts

    n_survives_severe = sum(
        1 for t in ticker_order
        if any(per_asset[t]["edge_extinguished_at"][str(lb)] == "survives_severe"
               for lb in LOOKBACKS))
    n_destroyed_base = base_counts["cost_destroyed"]

    if base_counts["cost_robust"] >= 5:
        class_verdict = "class_cost_robust"
    elif n_destroyed_base >= 8 or (n_destroyed_base >= 5 and n_survives_severe == 0):
        class_verdict = "class_cost_destroyed"
    else:
        class_verdict = "class_cost_sensitive"

    print("  base-cost cost-robustness verdict counts: "
          f"cost_robust={base_counts['cost_robust']} "
          f"(>=2 lookbacks) | cost_sensitive={base_counts['cost_sensitive']} "
          f"(1 lookback) | cost_destroyed={base_counts['cost_destroyed']}\n")
    print("  per-asset perturbation profiles (median log return, lookback order "
          "3/5/10, base cost):")
    for ticker in ticker_order:
        p = per_asset[ticker]
        meds = ", ".join(
            "{}:{:7.3f}/{:7.3f}/{:7.3f} {}".format(
                lb,
                p["lookbacks"][str(lb)]["base"]["medians"][0],
                p["lookbacks"][str(lb)]["stern"]["medians"][0],
                p["lookbacks"][str(lb)]["severe"]["medians"][0],
                p["lookbacks"][str(lb)]["base"]["verdict"],
            )
            for lb in LOOKBACKS
        )
        print("    {:6s} base={} rob={} surv={} "
              "extinguished={}".format(
            ticker, meds, p["cost_robustness"], p["edge_extinguished_at"],
            p["lookbacks"]["5"]["base"]["edge"]))
    print("")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {}
    for lb in LOOKBACKS:
        for t in target:
            for level, _c, _p in COST_LEVELS:
                res2[(lb, t, level)] = segment_result(t, tickers[t], lb,
                                                      cost_configs[level])
    out1 = json.dumps(
        {lb: {t: {level: per_asset[t]["lookbacks"][str(lb)][level]
                  for level, _c, _p in COST_LEVELS}
              for t in target}
         for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(
        {lb: {t: {level: res2[(lb, t, level)]
                  for level, _c, _p in COST_LEVELS}
              for t in target}
         for lb in LOOKBACKS}, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Base-cost snapshot for compact per-asset artifact view.
    def base_snapshot(ticker):
        p = per_asset[ticker]
        return {
            "lookback_medians": {str(lb): p["lookbacks"][str(lb)]["base"]["medians"]
                                 for lb in LOOKBACKS},
            "lookback_null_medians": {str(lb): p["lookbacks"][str(lb)]["base"]["null_medians"]
                                      for lb in LOOKBACKS},
            "verdicts_base": {str(lb): p["lookbacks"][str(lb)]["base"]["verdict"]
                              for lb in LOOKBACKS},
            "edges_base": {str(lb): p["lookbacks"][str(lb)]["base"]["edge"]
                           for lb in LOOKBACKS},
            "n_folds_base": {str(lb): p["lookbacks"][str(lb)]["base"]["n_folds"]
                             for lb in LOOKBACKS},
            "survival_grid": survival_grid,
            "cost_robustness": p["cost_robustness"],
            "edge_extinguished_at": p["edge_extinguished_at"],
        }

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_model=dict(
            description=(
                "slippage_cents + slippage_proportional (fraction of price); "
                "no commission_per_share (round-trip execution costs "
                "conservatively absorbed); target_exposure=1.0"
            ),
            levels={
                name: {"slippage_cents": c, "slippage_proportional": p}
                for name, c, p in COST_LEVELS
            },
        ),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            base_cost_verdict_counts=base_counts,
            survival_grid=survival_grid,
            n_survives_severe=n_survives_severe,
            n_destroyed_base_cost=n_destroyed_base,
            class_verdict=class_verdict,
            ticker_order=ticker_order,
            per_asset={t: base_snapshot(t) for t in ticker_order},
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("class-level verdict: {}".format(class_verdict))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Cost-sensitivity sweep of momentum lookback 3/5/10 on "
          "the collected 10-asset universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
