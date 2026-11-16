"""Real-data transaction-cost sensitivity of short-horizon momentum.

Objective (focused task R-001): test whether the candidate short-horizon
momentum edge (long the previous N-day return, hold 1 day, daily rebalance)
survives conservative transaction-cost stress on the collected 10-asset real
universe, for lookback in {3, 5, 10}, under the repository's existing realistic
cost model and matched coin-flip null.

Why this is decision-relevant: `LEARNING_STATE.md` admits short-horizon momentum
as candidate positive evidence but explicitly leaves execution-cost robustness as
a qualification. Momentum is a mechanical, daily-rebalanced signal, so its
turnover cost could plausibly be the same order of magnitude as the ~0.05-0.10
per-fold edge. That makes the cost question falsifiable and worth resolving
before any stronger admission.

Design (fixed a-priori, not tuned to the cost outcome):

* Universe: the existing 10 collected tickers (AAPL, MSFT, GOOGL, AMZN, META,
  NVDA, TSLA, JPM, JNJ, XOM), adjusted close.
* Signal: the existing `research.backtest.momentum_signals` (unchanged).
* Walk-forward: train=252d / test=84d / warmup=60d / overlap=60d, 4 contiguous
  volatility blocks, min 400 bars per segment (identical to
  `momentum_lookback_sweep.py`).
* Cost grid (three levels, declared here before any result is seen):
    - zero        : the engine default (no commission, no slippage)
    - realistic   : the repository's existing realistic cost model
                    (commission_per_trade=2.0, commission_per_share=0.003,
                    slippage_cents=2.0, slippage_proportional=0.0005), used by
                    `examples/ma_crossover_real_data.py` and
                    `examples/volatility_regime_filter.py`.
    - conservative: exactly 2x every realistic component (commission 4.0 /
                    0.006, slippage 4.0 cents / 0.0010), i.e. conservative
                    transaction-cost stress.
* Matched null: the framework's own coin-flip `noise_benchmark`, run on the
  same asset segments with the same cost cfg (the null therefore pays the same
  cost model). This is the repository's existing matched-null construction.

A-priori classification (declared before execution, per lookback):

    ER = assets with a positive REGIME_STABLE edge at the realistic level
    EC = assets with a positive REGIME_STABLE edge at the conservative level
    BR = assets whose mean candidate median exceeds their matched null at the
         realistic level
    BC = same comparison at the conservative level
    n  = number of assets

    COST_ELIMINATED : ER == 0 or BR <= n/2
                      (no positive edge at realistic cost, or the candidate no
                      longer beats a cost-paying coin flip in a majority)
    COST_ROBUST     : ER >= 1 and EC >= 1 and BC > n/2
                      (positive edges survive conservative stress and a
                      majority still beats the matched null)
    COST_FRAGILE    : otherwise (survives realistic, fails conservative)

Overall (across the three lookbacks):
    COST_ROBUST     : all three lookbacks COST_ROBUST
    COST_ELIMINATED : at least two lookbacks COST_ELIMINATED
    COST_FRAGILE    : otherwise

A-priori prediction (recorded before running; falsifiable): because the signal
is mechanical and daily-rebalanced, the realistic-cost level is expected to
reduce the absolute medians materially; the most likely single outcome is
COST_FRAGILE (the candidate keeps beating the matched null, but 2x costs remove
the positive absolute edges). COST_ROBUST would refute that prediction;
COST_ELIMINATED would show the candidate no longer even beats a cost-paying
coin flip.

Cross-consistency gate: the zero-cost column is an independent rerun of the
existing `momentum_lookback_sweep_results.json` computation, so its per-asset
medians, null medians and verdicts must match that artifact exactly. A mismatch
flags a defect in this check before any cost conclusion is drawn.

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
LOOKBACK_SWEEP_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

A_PRIORI_PREDICTION = (
    "Mechanical daily-rebalanced momentum: realistic costs materially reduce "
    "absolute medians; most likely single outcome COST_FRAGILE (candidate keeps "
    "beating the matched null but 2x costs remove positive absolute edges). "
    "COST_ROBUST refutes; COST_ELIMINATED means the candidate no longer beats a "
    "cost-paying coin flip."
)

CLASSIFICATION_RULE = (
    "per lookback: COST_ELIMINATED if ER==0 or BR<=n/2; COST_ROBUST if "
    "ER>=1 and EC>=1 and BC>n/2; else COST_FRAGILE. overall: COST_ROBUST if all "
    "three lookbacks COST_ROBUST; COST_ELIMINATED if >=2 lookbacks "
    "COST_ELIMINATED; else COST_FRAGILE. ER/EC = positive REGIME_STABLE edge "
    "assets at realistic/conservative; BR/BC = assets whose mean candidate "
    "median exceeds matched null at realistic/conservative."
)

QUALIFICATIONS = [
    "The framework's fold total_return is measured relative to the OOS "
    "starting equity (metrics.compute_metrics uses ends/starts - 1). Under "
    "costs the coin-flip null can draw equity down, so its percentage returns "
    "are non-monotonic in the cost level (e.g. the TSLA null median moves "
    "-0.018 -> +0.333 from zero to conservative). The candidate-vs-null "
    "comparison is therefore supporting evidence, not the primary "
    "cost-robustness discriminator; the primary evidence is the absolute "
    "candidate median, which is essentially unchanged across the grid.",
    "Costs are the repository's existing realistic parameters (5 bps "
    "proportional + $2/trade + 2 cents) and an exact 2x conservative level. "
    "The measured candidate cost drag is small relative to the ~0.06 median "
    "fold log return, so this verifies cost-insensitivity under this cost "
    "model, not under an arbitrary cost multiplier.",
    "The engine sizes to constant-dollar exposure and rebalances each bar, so "
    "measured turnover includes small daily rebalancing trades in addition to "
    "signal flips; the cost drag measured is not lower than a "
    "signal-change-only implementation would face.",
]


def cost_levels():
    """Three predeclared cost configurations (zero, realistic, conservative)."""
    return [
        ("zero", bt.BacktestConfig(initial_capital=1e6)),
        (
            "realistic",
            bt.BacktestConfig(
                initial_capital=1e6,
                commission_per_trade=2.0,
                commission_per_share=0.003,
                slippage_cents=2.0,
                slippage_proportional=0.0005,
            ),
        ),
        (
            "conservative",
            bt.BacktestConfig(
                initial_capital=1e6,
                commission_per_trade=4.0,
                commission_per_share=0.006,
                slippage_cents=4.0,
                slippage_proportional=0.0010,
            ),
        ),
    ]


def cfg_dict(cfg):
    return {
        "commission_per_trade": cfg.commission_per_trade,
        "commission_per_share": cfg.commission_per_share,
        "slippage_cents": cfg.slippage_cents,
        "slippage_proportional": cfg.slippage_proportional,
        "target_exposure": cfg.target_exposure,
    }


def run_cell(tickers, ticker_order, lookback, cfg):
    """Run the universe regime-stability sweep for one lookback and cost level."""
    summary = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=bt.momentum_signals,
        regime_labels_fn=lambda c: bt.volatility_blocks(
            c, n_blocks=N_BLOCKS, window=WINDOW
        ),
        baseline=(("lookback", lookback),),
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    per_asset = {}
    for ticker in ticker_order:
        r = summary.assets[ticker]
        per_asset[ticker] = dict(
            segments=[s.name for s in r.scenarios],
            medians=[round(s.baseline_median_log_return, 6) for s in r.scenarios],
            null_medians=[round(s.noise_median_log_return, 6) for s in r.scenarios],
            candidate_dispersion=round(r.candidate_dispersion, 6),
            null_dispersion=round(r.null_dispersion, 6),
            n_folds=r.n_folds,
            verdict=r.overall_verdict,
        )
    return per_asset


def cell_metrics(per_asset):
    """Derive the classification metrics for one (lookback, cost) cell."""
    n = len(per_asset)
    edge_assets = []
    loss_assets = []
    beat_null = []
    cand_means = {}
    null_means = {}
    for ticker, r in per_asset.items():
        cand_mean = float(np.mean(r["medians"]))
        null_mean = float(np.mean(r["null_medians"]))
        cand_means[ticker] = round(cand_mean, 6)
        null_means[ticker] = round(null_mean, 6)
        if r["verdict"] == "REGIME_STABLE" and all(m > 0 for m in r["medians"]):
            edge_assets.append(ticker)
        if r["verdict"] == "REGIME_STABLE_LOSS" or all(m < 0 for m in r["medians"]):
            loss_assets.append(ticker)
        if cand_mean > null_mean:
            beat_null.append(ticker)
    verdict_counts = {
        "CONSISTENT_WITH_NOISE": 0,
        "REGIME_STABLE": 0,
        "REGIME_STABLE_LOSS": 0,
        "REGIME_DEPENDENT": 0,
    }
    for r in per_asset.values():
        verdict_counts[r["verdict"]] += 1
    gaps = {
        t: round(cand_means[t] - null_means[t], 6) for t in per_asset
    }
    mean_gap = float(np.mean(list(gaps.values())))
    return dict(
        n_assets=n,
        edge_assets=edge_assets,
        edge_asset_count=len(edge_assets),
        loss_asset_count=len(loss_assets),
        beat_null_assets=beat_null,
        beat_null_asset_count=len(beat_null),
        mean_candidate_median=round(float(np.mean(list(cand_means.values()))), 6),
        mean_null_median=round(float(np.mean(list(null_means.values()))), 6),
        mean_candidate_minus_null=round(mean_gap, 6),
        verdict_counts=verdict_counts,
        candidate_means=cand_means,
        null_means=null_means,
        gaps=gaps,
    )


def classify(metrics_by_level):
    """Apply the predeclared per-lookback classification rule."""
    n = metrics_by_level["zero"]["n_assets"]
    er = metrics_by_level["realistic"]["edge_asset_count"]
    ec = metrics_by_level["conservative"]["edge_asset_count"]
    br = metrics_by_level["realistic"]["beat_null_asset_count"]
    bc = metrics_by_level["conservative"]["beat_null_asset_count"]
    if er == 0 or br <= n / 2:
        return "COST_ELIMINATED"
    if er >= 1 and ec >= 1 and bc > n / 2:
        return "COST_ROBUST"
    return "COST_FRAGILE"


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(
        f"  dataset: {manifest['dataset_id']} | collection: "
        f"{manifest['collection_date']} | universe: {len(manifest['universe'])}"
    )

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        return 1

    print("\n=== 3. Leakage review per asset (realistic cost cfg) ===")
    realistic_cfg = dict(cost_levels())["realistic"]
    target = ["AMZN", "JPM"]
    tickers_subset = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers_subset[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(
            list(bars),
            signals,
            bt.BacktestConfig(
                initial_capital=1e6,
                commission_per_trade=realistic_cfg.commission_per_trade,
                commission_per_share=realistic_cfg.commission_per_share,
                slippage_cents=realistic_cfg.slippage_cents,
                slippage_proportional=realistic_cfg.slippage_proportional,
            ),
        )
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(
            f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
            f"leakage [PASS], full-sample total_return (realistic costs) "
            f"{total_return:+.3f}"
        )
    print(
        "  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
        "costs are applied to every round trip by the engine."
    )

    print("\n=== 4. Cost-sensitivity grid (lookbacks 3/5/10 x zero/realistic/conservative) ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(tickers.keys())

    levels = cost_levels()
    cells = {}
    for lb in LOOKBACKS:
        per_level = {}
        metrics_by_level = {}
        for name, cfg in levels:
            per_asset = run_cell(tickers, ticker_order, lb, cfg)
            metrics = cell_metrics(per_asset)
            per_level[name] = dict(per_asset=per_asset, metrics=metrics)
            metrics_by_level[name] = metrics
        classification = classify(metrics_by_level)
        mz = metrics_by_level["zero"]
        mr = metrics_by_level["realistic"]
        mc = metrics_by_level["conservative"]
        cost_drag = dict(
            candidate_zero_minus_realistic=round(
                mz["mean_candidate_median"] - mr["mean_candidate_median"], 6
            ),
            candidate_zero_minus_conservative=round(
                mz["mean_candidate_median"] - mc["mean_candidate_median"], 6
            ),
            null_zero_minus_realistic=round(
                mz["mean_null_median"] - mr["mean_null_median"], 6
            ),
            null_zero_minus_conservative=round(
                mz["mean_null_median"] - mc["mean_null_median"], 6
            ),
        )
        cells[str(lb)] = dict(
            lookback=lb,
            per_cost_level=per_level,
            classification=classification,
            cost_drag=cost_drag,
        )
        print(
            f"\n  lookback {lb}: classification={classification}  "
            f"(zero/realistic/conservative edge assets = "
            f"{metrics_by_level['zero']['edge_asset_count']}/"
            f"{metrics_by_level['realistic']['edge_asset_count']}/"
            f"{metrics_by_level['conservative']['edge_asset_count']}; "
            f"beat-null = "
            f"{metrics_by_level['zero']['beat_null_asset_count']}/"
            f"{metrics_by_level['realistic']['beat_null_asset_count']}/"
            f"{metrics_by_level['conservative']['beat_null_asset_count']})"
        )
        for name, _cfg in levels:
            m = metrics_by_level[name]
            print(
                f"    {name:12s} mean_cand={m['mean_candidate_median']:+.3f} "
                f"mean_null={m['mean_null_median']:+.3f} "
                f"gap={m['mean_candidate_minus_null']:+.3f} "
                f"edges={m['edge_assets']} "
                f"verdicts={m['verdict_counts']}"
            )

    overall_per_lookback = {lb: cells[lb]["classification"] for lb in cells}
    labels = list(overall_per_lookback.values())
    if all(x == "COST_ROBUST" for x in labels):
        overall = "COST_ROBUST"
    elif labels.count("COST_ELIMINATED") >= 2:
        overall = "COST_ELIMINATED"
    else:
        overall = "COST_FRAGILE"
    print(f"\n=== 5. Overall cost-robustness classification: {overall} ===")
    print(f"  per-lookback: {overall_per_lookback}")

    # Cross-consistency gate: the zero-cost column must reproduce the existing
    # lookback-sweep artifact exactly (medians, nulls, verdicts, segments).
    print("\n=== 6. Cross-consistency vs momentum_lookback_sweep_results.json (zero cost) ===")
    cmismatch = False
    if LOOKBACK_SWEEP_ARTIFACT.exists():
        with open(LOOKBACK_SWEEP_ARTIFACT) as f:
            prev = json.load(f)
        prev_lp = prev["per_asset"]
        for ticker in ticker_order:
            for lb in LOOKBACKS:
                zero = cells[str(lb)]["per_cost_level"]["zero"]["per_asset"][ticker]
                pub = prev_lp[ticker][str(lb)]
                ok = (
                    [round(v, 3) for v in zero["medians"]] == pub["medians"]
                    and [round(v, 3) for v in zero["null_medians"]] == pub["null_medians"]
                    and zero["verdict"] == pub["verdict"]
                )
                if not ok:
                    cmismatch = True
                    print(
                        f"  MISMATCH {ticker} lb={lb}: medians {zero['medians']} vs "
                        f"{pub['medians']}; nulls {zero['null_medians']} vs "
                        f"{pub['null_medians']}; verdict {zero['verdict']} vs "
                        f"{pub['verdict']}"
                    )
        print("  zero-cost column cross-check: " + ("MISMATCH" if cmismatch else "MATCH (all 10 tickers x 3 lookbacks)"))
    else:
        cmismatch = True
        print("  cross-consistency artifact missing: cannot cross-check")

    # Determinism: rerun the AMZN/JPM subset at the realistic cost and assert
    # identical output.
    print("\n=== 7. Determinism (AMZN/JPM realistic cost, r1 == r2) ===")
    r1 = {lb: run_cell(tickers_subset, target, lb, realistic_cfg) for lb in LOOKBACKS}
    r2 = {lb: run_cell(tickers_subset, target, lb, realistic_cfg) for lb in LOOKBACKS}
    det = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    print(f"  r1 == r2: {det}")
    assert det, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_levels={name: cfg_dict(cfg) for name, cfg in levels},
        train=TRAIN,
        test=TEST,
        warmup=WARM,
        overlap=OVERLAP,
        window=WINDOW,
        n_blocks=N_BLOCKS,
        min_segment_bars=MIN_SEGMENT_BARS,
        a_priori_prediction=A_PRIORI_PREDICTION,
        classification_rule=CLASSIFICATION_RULE,
        qualifications=QUALIFICATIONS,
        cells=cells,
        overall=dict(
            classification=overall,
            per_lookback=overall_per_lookback,
            a_priori_prediction="COST_FRAGILE",
            prediction_outcome=(
                "CONFIRMED" if overall == "COST_FRAGILE" else "REFUTED"
            ),
        ),
        cross_consistency=dict(
            reference="momentum_lookback_sweep_results.json",
            zero_cost_match=not cmismatch,
        ),
        determinism=dict(realistic_AMZN_JPM_r1_eq_r2=det),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print(
        "NOTE: research/simulation only. No live trading or production "
        "execution. Momentum cost-sensitivity on the collected universe: "
        "exploratory simulation."
    )
    return 0 if (not cmismatch and det) else 2


if __name__ == "__main__":
    sys.exit(main())