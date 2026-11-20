"""Real-data transaction-cost sensitivity of the short-horizon momentum edge.

Objective (task R-001): the momentum class (long the previous N-day return,
hold 1 day, daily rebalance) is the only admitted candidate positive evidence
(`research/checks/momentum.py`, `momentum_lookback_sweep.py`). That admission is
conditional on the edge being decision-relevant, which requires it to survive
execution costs. This check runs a *bounded, predeclared* cost grid for
lookbacks 3 / 5 / 10 across the 10-asset collected universe through the existing
walk-forward + regime-stability + matched coin-flip null stack, and classifies
whether the edge is cost-robust.

Cost model (the repository's existing "realistic costs" profile, used verbatim
in `examples/ma_crossover_real_data.py` and `examples/volatility_regime_filter.py`):
    commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005 (5 bps per side).
The grid is a fixed monotone ladder around that profile, declared before any
result is observed:

    zero            : 0 cost (reference; must reproduce the prior lookback sweep)
    repo_realistic  : 1x the existing realistic profile
    conservative_2x : 2x every cost component
    severe_4x       : 4x every cost component
    extreme_10x     : 10x every cost component (breaking-point probe)

Matched null: for every (asset, lookback, cost) cell the candidate is compared
against a coin-flip null run on the same segments *with the same cost config*
(`noise_benchmark` inside `stress_segments` receives the same `cfg`). Both the
candidate and the null therefore pay identical costs, so the candidate-minus-null
gap isolates the net edge rather than the gross edge.

A-priori prediction (declared before execution, falsifiable):
  Costs are a pure drag and the signal is daily rebalanced, so candidate medians
  are predicted to be non-increasing in the cost level for every asset. Because
  the constant-dollar rebalance produces small per-bar position changes and the
  sign flips are infrequent, the drag is predicted to be a fraction of the
  per-segment medians. Headline prediction: COST_ROBUST — the conservative_2x
  stress retains at least half of the zero-cost robust assets with a positive
  candidate-minus-null gap. The severe_4x / extreme_10x columns locate the
  breaking point if the prediction is wrong.

Classification rule (predeclared, evaluated on conservative_2x):
  EDGE(asset, cost)        = the asset's lookback has verdict REGIME_STABLE and
                             every segment median > 0.
  robust_assets(cost)      = assets with EDGE at >= 2 of the 3 lookbacks.
  E(cost)                  = |robust_assets(cost)|.
  gap(cost)                = median(all candidate segment medians)
                             - median(all null segment medians).
  COST_ROBUST  : E(conservative_2x) >= max(2, ceil(E(zero)/2)) and gap(conservative_2x) > 0
  COST_FRAGILE : 1 <= E(conservative_2x) < that threshold, or gap(conservative_2x) <= 0
  COST_ERASED  : E(conservative_2x) == 0

Method (fixed, not tuned to OOS): momentum lookbacks 3 / 5 / 10; walk-forward
train=252d / test=84d / warmup=60d / overlap=60d; 4 contiguous date blocks with
a past-only volatility label; min 400 bars/segment. A cross-consistency gate
asserts the zero-cost column reproduces `momentum_lookback_sweep_results.json`
exactly (medians, nulls, verdicts), and determinism is asserted by an internal
recomputation of the AMZN/JPM subset.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
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

# The repository's existing realistic cost profile, verbatim from
# examples/ma_crossover_real_data.py and examples/volatility_regime_filter.py.
_REPO_REALISTIC = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


def _scaled(mult: float) -> dict:
    return {k: v * mult for k, v in _REPO_REALISTIC.items()}


# Predeclared cost grid: (name, cost kwargs). Monotone in every component.
COST_CONFIGS = [
    ("zero", dict(commission_per_trade=0.0, commission_per_share=0.0,
                  slippage_cents=0.0, slippage_proportional=0.0)),
    ("repo_realistic", dict(_REPO_REALISTIC)),
    ("conservative_2x", _scaled(2.0)),
    ("severe_4x", _scaled(4.0)),
    ("extreme_10x", _scaled(10.0)),
]

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
PRIOR_SWEEP_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def make_cfg(cost: dict) -> bt.BacktestConfig:
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=0,
                             periods_per_year=252, **cost)


def lookback_verdict(sr: dict) -> str:
    """EDGE / LOSS / NO_EDGE for one asset-lookback segment result.

    Mirrors `research/checks/momentum_lookback_sweep.py::lookback_verdict`.
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(edge_count: int) -> str:
    """ROBUST (>=2 lookbacks with a positive edge) / SENSITIVE (1) / NO_EDGE (0)."""
    if edge_count >= 2:
        return "ROBUST"
    if edge_count == 1:
        return "SENSITIVE"
    return "NO_EDGE"


def segment_result(ticker, bars, lookback, cost):
    """Walk-forward regime-stability result for one ticker/lookback/cost cell."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=list(bars),
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=make_cfg(cost),
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


def summarize_asset(cost_name, lookback_results):
    """Reduce one asset's lookback results at one cost level to a profile."""
    edge_count = sum(1 for lb in LOOKBACKS
                     if lookback_verdict(lookback_results[str(lb)]) == "EDGE")
    all_cand = [m for lb in LOOKBACKS for m in lookback_results[str(lb)]["medians"]]
    all_null = [m for lb in LOOKBACKS for m in lookback_results[str(lb)]["null_medians"]]
    return dict(
        cost=cost_name,
        lookbacks={str(lb): lookback_results[str(lb)] for lb in LOOKBACKS},
        edge_count=edge_count,
        robustness_verdict=robustness_verdict(edge_count),
        candidate_median_all=round(float(np.median(all_cand)), 3),
        null_median_all=round(float(np.median(all_null)), 3),
        candidate_mean_all=round(float(np.mean(all_cand)), 3),
        null_mean_all=round(float(np.mean(all_null)), 3),
    )


def classify(universe_cost_summary):
    """Apply the predeclared cost-robustness classification rule."""
    e0 = universe_cost_summary["zero"]["n_robust_assets"]
    e1 = universe_cost_summary["conservative_2x"]["n_robust_assets"]
    threshold = max(2, int(math.ceil(e0 / 2.0)))
    gap1 = universe_cost_summary["conservative_2x"]["median_gap"]
    if e1 >= threshold and gap1 > 0:
        headline = "COST_ROBUST"
    elif e1 >= 1:
        headline = "COST_FRAGILE"
    else:
        headline = "COST_ERASED"
    return dict(
        headline=headline,
        zero_cost_robust_assets=e0,
        conservative_robust_assets=e1,
        required_conservative_assets=threshold,
        conservative_gap=round(gap1, 3),
        prediction="COST_ROBUST",
        prediction_confirmed=(headline == "COST_ROBUST"),
    )


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
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

    print("\n=== 4. Cost grid (predeclared) ===")
    for name, cost in COST_CONFIGS:
        print(f"  {name:16s} {cost}")
    print("  walk-forward train={}d / test={}d / warmup={}d / overlap={}d, "
          "4 volatility blocks, min {} bars/segment, lookbacks {}\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, list(LOOKBACKS)))

    # Full universe: results[cost_name][ticker][lookback].
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    results = {}
    for cost_name, cost in COST_CONFIGS:
        results[cost_name] = {}
        for ticker in ticker_order:
            per_lb = {str(lb): segment_result(ticker, all_tickers[ticker], lb, cost)
                      for lb in LOOKBACKS}
            results[cost_name][ticker] = summarize_asset(cost_name, per_lb)

    # Universe-level summary per cost level.
    cost_summary = {}
    for cost_name, _ in COST_CONFIGS:
        robust_assets = sorted(t for t in ticker_order
                               if results[cost_name][t]["robustness_verdict"] == "ROBUST")
        edge_assets = sorted(t for t in ticker_order
                             if results[cost_name][t]["edge_count"] >= 1)
        all_cand = [m for t in ticker_order for lb in LOOKBACKS
                    for m in results[cost_name][t]["lookbacks"][str(lb)]["medians"]]
        all_null = [m for t in ticker_order for lb in LOOKBACKS
                    for m in results[cost_name][t]["lookbacks"][str(lb)]["null_medians"]]
        cand_med = float(np.median(all_cand))
        null_med = float(np.median(all_null))
        gaps = [c - n for t in ticker_order for lb in LOOKBACKS
                for c, n in zip(results[cost_name][t]["lookbacks"][str(lb)]["medians"],
                                results[cost_name][t]["lookbacks"][str(lb)]["null_medians"])]
        verdict_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
        for t in ticker_order:
            verdict_counts[results[cost_name][t]["robustness_verdict"]] += 1
        cost_summary[cost_name] = dict(
            cost=cost_name,
            robust_assets=robust_assets,
            edge_assets=edge_assets,
            n_robust_assets=len(robust_assets),
            n_edge_assets=len(edge_assets),
            median_candidate=round(cand_med, 3),
            median_null=round(null_med, 3),
            median_gap=round(cand_med - null_med, 3),
            mean_gap=round(float(np.mean(gaps)), 3),
            robustness_verdict_counts=verdict_counts,
            n_segments=len(all_cand),
        )

    print("=== 5. Universe summary by cost level ===")
    print("  {:16s} {:>8s} {:>8s} {:>9s} {:>9s}  robust assets".format(
        "cost", "robust", "edge", "med_cand", "med_gap"))
    for cost_name, _ in COST_CONFIGS:
        s = cost_summary[cost_name]
        print("  {:16s} {:>8d} {:>8d} {:>+9.3f} {:>+9.3f}  {}".format(
            cost_name, s["n_robust_assets"], s["n_edge_assets"],
            s["median_candidate"], s["median_gap"], ",".join(s["robust_assets"]) or "-"))
    print("")

    print("=== 6. Per-asset cost-sensitivity profile ===")
    print("  {:6s} {:>16s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "tkr", "robust@cost", "med@zero", "med@real", "med@2x", "med@4x", "med@10x"))
    for ticker in ticker_order:
        meds = []
        for cost_name, _ in COST_CONFIGS:
            lb = results[cost_name][ticker]["lookbacks"]["5"]
            meds.append(float(np.median(lb["medians"])))
        rob = "z{}/r{}/2x{}/4x{}/10x{}".format(
            results["zero"][ticker]["robustness_verdict"][0],
            results["repo_realistic"][ticker]["robustness_verdict"][0],
            results["conservative_2x"][ticker]["robustness_verdict"][0],
            results["severe_4x"][ticker]["robustness_verdict"][0],
            results["extreme_10x"][ticker]["robustness_verdict"][0])
        print("  {:6s} {:>16s} {:>+12.3f} {:>+12.3f} {:>+12.3f} {:>+12.3f} {:>+12.3f}".format(
            ticker, rob, meds[0], meds[1], meds[2], meds[3], meds[4]))
    print("")

    # Cross-consistency gate: the zero-cost column must reproduce the prior
    # momentum_lookback_sweep_results.json exactly (medians, nulls, verdicts).
    print("=== 7. Cross-consistency: zero-cost column vs prior lookback sweep ===")
    with open(PRIOR_SWEEP_PATH) as f:
        prior = json.load(f)
    xcheck_ok = True
    for ticker in ticker_order:
        for lb in LOOKBACKS:
            pub = prior["per_asset"][ticker][str(lb)]
            got = results["zero"][ticker]["lookbacks"][str(lb)]
            same = (got["medians"] == pub["medians"]
                    and got["null_medians"] == pub["null_medians"]
                    and got["verdict"] == pub["verdict"])
            if not same:
                xcheck_ok = False
                print("  MISMATCH {} lookback {}: got {} / {} / {} vs prior {} / {} / {}".format(
                    ticker, lb, got["medians"], got["null_medians"], got["verdict"],
                    pub["medians"], pub["null_medians"], pub["verdict"]))
    print("  zero-cost column reproduces prior lookback sweep: {}".format(xcheck_ok))
    if not xcheck_ok:
        print("CROSS-CONSISTENCY FAILED; aborting before writing the artifact.")
        sys.exit(1)
    print("")

    # Determinism: rerun the target subset at every cost level and assert
    # identical output to the full-universe run.
    print("=== 8. Determinism (AMZN/JPM rerun at all cost levels) ===")
    rerun = {}
    for cost_name, cost in COST_CONFIGS:
        rerun[cost_name] = {}
        for ticker in target:
            per_lb = {str(lb): segment_result(ticker, tickers[ticker], lb, cost)
                      for lb in LOOKBACKS}
            rerun[cost_name][ticker] = summarize_asset(cost_name, per_lb)
    out1 = json.dumps({c: {t: results[c][t] for t in target} for c, _ in COST_CONFIGS},
                      sort_keys=True)
    out2 = json.dumps(rerun, sort_keys=True)
    print("  determinism r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    headline = classify(cost_summary)
    print("\n=== 9. Cost-robustness classification (predeclared rule) ===")
    print("  headline: {} (prediction {} -> {})".format(
        headline["headline"], headline["prediction"],
        "CONFIRMED" if headline["prediction_confirmed"] else "REJECTED"))
    print("  zero-cost robust assets: {} | conservative_2x robust assets: {} "
          "(required >= {}) | conservative gap {:+.3f}".format(
              headline["zero_cost_robust_assets"],
              headline["conservative_robust_assets"],
              headline["required_conservative_assets"],
              headline["conservative_gap"]))
    print("")

    finished = datetime.now(timezone.utc).isoformat()
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        started_at_utc=started,
        finished_at_utc=finished,
        cost_configs=[dict(name=n, cost=c) for n, c in COST_CONFIGS],
        repo_realistic_profile=dict(_REPO_REALISTIC),
        classification_rule=dict(
            edge="verdict REGIME_STABLE and all segment medians > 0",
            robust_assets="assets with EDGE at >= 2 of the 3 lookbacks",
            gap="median(all candidate segment medians) - median(all null segment medians)",
            headline=(
                "COST_ROBUST if E(conservative_2x) >= max(2, ceil(E(zero)/2)) and "
                "gap(conservative_2x) > 0; COST_FRAGILE if 1 <= E(conservative_2x) < "
                "that threshold or gap <= 0; COST_ERASED if E(conservative_2x) == 0"),
        ),
        a_priori_prediction=(
            "Candidate medians are non-increasing in cost; COST_ROBUST at "
            "conservative_2x (>= half the zero-cost robust assets, positive "
            "candidate-minus-null gap). severe_4x / extreme_10x locate the "
            "breaking point."),
        per_asset=results,
        universe=dict(
            cost_summary=cost_summary,
            headline=headline,
            ticker_order=ticker_order,
            zero_cost_crosscheck_passed=xcheck_ok,
        ),
        determinism=dict(passed=True, scope="AMZN/JPM rerun at all cost levels"),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum transaction-cost sensitivity: exploratory "
          "simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())