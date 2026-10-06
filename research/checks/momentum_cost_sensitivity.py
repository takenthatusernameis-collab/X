#!/usr/bin/env python3
"""Cost-sensitivity check for short-horizon momentum lookback 3/5/10.

Primary question: does the candidate positive momentum edge (long the previous
lookback-day return, hold 1 day, daily rebalanced) survive conservative
transaction-cost stress on the collected 10-asset adjusted-close universe?

Method: walk-forward regime-stability gate (train=252d/test=84d/warmup=60d/
overlap=60d; 4 contiguous volatility blocks, min 400 bars/segment) with a
matched coin-flip null, run under three cost levels for each lookback:
  - zero           : no costs (baseline reference)
  - realistic      : the repository's realistic cost model, identical to
                     examples/ma_crossover.py:
                     commission_per_trade=2.0, commission_per_share=0.003,
                     slippage_cents=2.0, slippage_proportional=0.0005
  - stressed       : 2x realistic
Grid: 3 lookbacks x 3 cost levels = 9 configs over the collected universe.

The coin-flip null benchmark is computed inside stress_segments with the SAME
BacktestConfig as the candidate, so transaction costs are applied identically
to candidate and null (matched null). Cost impact is reported as the difference
in segment median log returns between cost levels.

Cost-robustness classification (a-priori deterministic rule, applied only to
published medians and verdicts — no tuning to any other observed quantity).
An asset is an "edge asset" at a cost level if its overall verdict is
REGIME_STABLE and every segment median is strictly positive. For each lookback:
  - ERODED   : >= half of the zero-cost edge assets lose their edge at
               realistic costs (verdict changes, a median falls <= 0, or a
               median falls inside the 0.05 noise band);
               or the mean relative median erosion across remaining edge
               assets at realistic costs is >= 50%.
  - ROBUST   : no zero-cost edge asset loses its edge at realistic costs AND
               the mean relative median erosion across edge assets at
               realistic costs is < 25%.
  - SENSITIVE: all other cases (erosion >= 25% and/or some edge assets
               weakened without reaching ERODED; or edges survive realistic
               but are lost at stressed costs).

A loss of edge at stressed costs alone does not make the lookback ERODED at
realistic cost; it is reported in the per-level detail.

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
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

ZERO_CFG = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
REALISTIC_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
STRESSED_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=4.0,
    commission_per_share=0.006,
    slippage_cents=4.0,
    slippage_proportional=0.001,
)
COST_LEVELS = {"zero": ZERO_CFG, "realistic": REALISTIC_CFG, "stressed": STRESSED_CFG}
NOISE_BAND = 0.05
TARGET_ASSETS = ("AMZN", "JPM")


def serialize_result(res, label):
    """Serialize a RegimeStressResult to a JSON-plain record."""
    return {
        "label": label,
        "n_folds": res.n_folds,
        "n_periods": res.n_periods,
        "train_window_bars": res.train_window_bars,
        "test_window_bars": res.test_window_bars,
        "scenario_names": [s.name for s in res.scenarios],
        "candidate_medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(res.null_dispersion, 3),
        "verdict": res.overall_verdict,
        "baseline_param_set": [tuple(p) for p in res.baseline_param_set],
    }


def classify_cost_robustness(zero_assets, real_assets, stressed_assets):
    """Apply the a-priori deterministic classification rule.

    Edge asset: overall_verdict == REGIME_STABLE and all scenario medians > 0.
    Uses only medians and verdicts; no other observed quantity.
    """
    zero_edge = [
        t
        for t in zero_assets
        if zero_assets[t].overall_verdict == "REGIME_STABLE"
        and all(s.baseline_median_log_return > 0.0 for s in zero_assets[t].scenarios)
    ]
    n_edge = len(zero_edge)
    if n_edge == 0:
        return dict(
            classification="ERODED",
            zero_cost_edge_assets=[],
            zero_cost_edge_count=0,
            edge_assets_lost_at_realistic=[],
            edge_assets_lost_count=0,
            mean_relative_median_erosion_realistic=0.0,
            edge_assets_lost_at_stressed=[],
        )
    real_lost = [
        t
        for t in zero_edge
        if real_assets[t].overall_verdict != "REGIME_STABLE"
        or any(s.baseline_median_log_return <= 0.0 for s in real_assets[t].scenarios)
        or any(
            abs(s.baseline_median_log_return - n) <= NOISE_BAND
            for s, n in zip(real_assets[t].scenarios, real_assets[t].noise_medians)
        )
    ]
    stressed_lost = [
        t
        for t in zero_edge
        if stressed_assets[t].overall_verdict != "REGIME_STABLE"
        or any(s.baseline_median_log_return <= 0.0 for s in stressed_assets[t].scenarios)
        or any(
            abs(s.baseline_median_log_return - n) <= NOISE_BAND
            for s, n in zip(stressed_assets[t].scenarios, stressed_assets[t].noise_medians)
        )
    ]
    remaining = [t for t in zero_edge if t not in real_lost]
    erosi = []
    for t in remaining:
        z = np.mean([s.baseline_median_log_return for s in zero_assets[t].scenarios])
        r = np.mean([s.baseline_median_log_return for s in real_assets[t].scenarios])
        erosi.append((r - z) / z if z != 0 else 0.0)
    mean_er = float(np.mean(erosi)) if erosi else 0.0
    if len(real_lost) >= n_edge / 2 or mean_er <= -0.5:
        base = "ERODED"
    elif len(real_lost) == 0 and mean_er > -0.25:
        base = "ROBUST"
    else:
        base = "SENSITIVE"
    return dict(
        classification=base,
        zero_cost_edge_assets=zero_edge,
        zero_cost_edge_count=n_edge,
        edge_assets_lost_at_realistic=real_lost,
        edge_assets_lost_count=len(real_lost),
        edge_assets_lost_at_stressed=stressed_lost,
        edge_assets_lost_at_stressed_count=len(stressed_lost),
        mean_relative_median_erosion_realistic=round(mean_er, 3),
    )


def load_universe():
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    return manifest, tickers


def run_universe(tickers, lookback, cfg):
    baseline = (("lookback", lookback),)
    param_grid = [{"lookback": lookback}]
    return bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=baseline,
        param_grid=param_grid,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg,
    )


def run_asset(lookback, cost_label, cfg, asset):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=signals,
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
    return res


def per_asset_detail(lookback, cfg, asset):
    res = run_asset(lookback, "x", cfg, asset)
    return {
        "label": res.scenarios[0].name,
        "n_folds": res.n_folds,
        "scenario_names": [s.name for s in res.scenarios],
        "candidate_medians": [round(s.baseline_median_log_return, 3) for s in res.scenarios],
        "null_medians": [round(s.noise_median_log_return, 3) for s in res.scenarios],
        "candidate_dispersion": round(res.candidate_dispersion, 3),
        "null_dispersion": round(res.null_dispersion, 3),
        "verdict": res.overall_verdict,
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, tickers = load_universe()

    # --- 1. Manifest integrity ---
    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(
        f"  dataset: {manifest['dataset_id']} | collection: "
        f"{manifest['collection_date']} | universe: {len(manifest['universe'])}"
    )

    # --- 2. Data preflight ---
    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(
        0,
        str(
            Path(__file__).resolve().parent.parent.parent / "research" / "data"
        ),
    )
    from importlib import import_module

    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting.")
        return 1

    # --- 3. Leakage review on AMZN / JPM (both cost levels, lookback 5) ---
    print("\n=== 3. Leakage review per asset (lookback 5, zero & realistic) ===")
    for asset in TARGET_ASSETS:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        for label, cfg in (("zero", ZERO_CFG), ("realistic", REALISTIC_CFG)):
            signals = bt.momentum_signals(closes, lookback=5)
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
            res = bt.run_bars(list(bars), signals, cfg)
            bt.check_equity_matches_fills(
                res.equity_curve, res.trades, closes, cfg.initial_capital
            )
            print(
                f"  {asset} [{label}]: {bars.n_bars} bars, "
                f"leakage [PASS], trades={res.trades[-3:] if res.trades else 0}"
            )
    print(
        "  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
        "neutral bars before the lookback window carry the equity forward."
    )

    # --- 4. Run the cost grid: 9 configs x full universe ---
    print("\n=== 4. Cost grid: lookback 3/5/10 x zero/realistic/stressed (full universe) ===")
    results = {}
    for lookback in LOOKBACKS:
        for level, cfg in COST_LEVELS.items():
            print(f"  [{lookback}d, {level}] running...")
            results[(lookback, level)] = run_universe(tickers, lookback, cfg)
            res = results[(lookback, level)]
            counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0, "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
            for a in res.assets.values():
                counts[a.overall_verdict] += 1
            cands = [s.baseline_median_log_return for a in res.assets.values() for s in a.scenarios]
            nulls = [s.noise_median_log_return for a in res.assets.values() for s in a.scenarios]
            print(
                f"    {level}: verdicts "
                f"{', '.join(f'{k}={v}' for k, v in counts.items())}, "
                f"candidate_disp={np.std(cands):+.3f}, "
                f"null_disp={np.std(nulls):+.3f}"
            )

    # --- 5. AMZN / JPM detailed runs at all 9 configs ---
    print("\n=== 5. AMZN / JPM detailed (all 9 configs) ===")
    detail = {}
    for lookback in LOOKBACKS:
        for level, cfg in COST_LEVELS.items():
            for asset in TARGET_ASSETS:
                detail[(asset, lookback, level)] = per_asset_detail(lookback, cfg, asset)
                res = detail[(asset, lookback, level)]
                print(
                    f"  {asset} [{lookback}d/{level}]: medians "
                    f"{res['candidate_medians']} null {res['null_medians']} "
                    f"verdict={res['verdict']}"
                )

    # --- 6. Cross-check: lookback=5 zero vs existing momentum_results.json ---
    print("\n=== 6. Cross-check lookback=5 zero-cost vs momentum_results.json ===")
    old = json.load(open(ARTIFACT_DIR / "momentum_results.json"))
    new = results[(5, "zero")]
    # compare universe-level per-asset medians and verdicts
    old_univ = old["universe"]["per_asset"]
    new_univ = {t: serialize_result(new.assets[t], "") for t in old_univ}
    cross_ok = True
    for ticker in old_univ:
        o_med = round(old_univ[ticker]["medians"][1], 3)
        n_med = new_univ[ticker]["candidate_medians"][1]
        o_ver = old_univ[ticker]["verdict"]
        n_ver = new_univ[ticker]["verdict"]
        ok = (o_med == n_med) and (o_ver == n_ver)
        cross_ok = cross_ok and ok
        print(f"  {ticker}: median {o_med} vs {n_med}, verdict {o_ver}/{n_ver} -> "
              f"{'OK' if ok else 'DIFF'}")
    print("  cross-check:", "MATCH" if cross_ok else "DIFF")

    # --- 7. Cost impact and classification ---
    print("\n=== 7. Cost impact & robustness classification ===")
    universe = {}
    for lookback in LOOKBACKS:
        universe[str(lookback)] = {
            level: serialize_universe_result(results[(lookback, level)], level)
            for level in COST_LEVELS
        }
    impact = {}
    for ticker in manifest["universe"]:
        impact[ticker] = {}
        for lookback in LOOKBACKS:
            z_med = [
                s.baseline_median_log_return for s in results[(lookback, "zero")].assets[ticker].scenarios
            ]
            r_med = [
                s.baseline_median_log_return for s in results[(lookback, "realistic")].assets[ticker].scenarios
            ]
            s_med = [
                s.baseline_median_log_return for s in results[(lookback, "stressed")].assets[ticker].scenarios
            ]
            impact[ticker][str(lookback)] = dict(
                zero=round(np.mean(z_med), 3),
                realistic=round(np.mean(r_med), 3),
                stressed=round(np.mean(s_med), 3),
                realistic_vs_zero=round(np.mean(r_med) - np.mean(z_med), 3),
                realistic_relative=round((np.mean(r_med) - np.mean(z_med)) / np.mean(z_med), 3)
                if np.mean(z_med) != 0
                else 0.0,
                stressed_vs_zero=round(np.mean(s_med) - np.mean(z_med), 3),
                stressed_relative=round((np.mean(s_med) - np.mean(z_med)) / np.mean(z_med), 3)
                if np.mean(z_med) != 0
                else 0.0,
            )
    classification = {}
    for lookback in LOOKBACKS:
        cl = classify_cost_robustness(
            results[(lookback, "zero")].assets,
            results[(lookback, "realistic")].assets,
            results[(lookback, "stressed")].assets,
        )
        classification[str(lookback)] = cl
        print(
            f"  lookback {lookback}d: {cl['zero_cost_edge_count']} zero-cost edge "
            f"assets -> {cl['edge_assets_lost_count']} lost at realistic, "
            f"mean relative erosion {cl['mean_relative_median_erosion_realistic']:+.1%} "
            f"-> {cl['classification']}"
        )

    # --- 8. Synthetic perturbation sweep (tooling check, zero-cost reference,
    #     unchanged from the momentum check) ---
    print("\n=== 8. Synthetic perturbation sweep (lookback 3/5/10, zero cost) ===")
    bars = bt.generate_bars(600, seed=SEED)
    grid = bt.parameter_grid_around((("lookback", 5),), multipliers=(0.5, 1.0, 2.0))
    sweep = bt.parameter_sweep(
        signals_fn=lambda c, **p: bt.momentum_signals(c, lookback=p["lookback"]),
        bars=list(bars),
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        cfg=ZERO_CFG,
        periods_per_year=252,
    )
    summary = bt.sweep_summary(sweep, baseline=(("lookback", 5),))
    noise = bt.noise_benchmark(
        bars=list(bars),
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        cfg=ZERO_CFG,
        periods_per_year=252,
        seed=SEED,
    )
    perturbation = dict(
        param_sets=[tuple(ps) for ps in sweep.param_sets],
        medians=[round(m, 3) for m in summary.median_log_returns],
        baseline_median=round(summary.baseline_median_log_return, 3),
        noise_median=round(noise.baseline_median_log_return, 3),
        compare_noise_default=summary.compare_noise(noise.baseline_median_log_return),
        compare_noise_fixed=summary.compare_noise(noise.baseline_median_log_return, tol=0.05),
        effective_tolerance=round(summary.effective_tolerance, 4),
        n_folds=summary.n_folds,
    )
    print(f"  param sets: {perturbation['param_sets']}")
    print(f"  medians: {perturbation['medians']}")
    print(f"  baseline: {perturbation['baseline_median']:+.3f} | null: "
          f"{perturbation['noise_median']:+.3f}")

    # --- 9. Determinism: rerun all 9 configs and assert identical output ---
    print("\n=== 9. Determinism (rerun all 9 configs) ===")
    results2 = {}
    for lookback in LOOKBACKS:
        for level, cfg in COST_LEVELS.items():
            results2[(lookback, level)] = run_universe(tickers, lookback, cfg)
    out1 = json.dumps(
        {
            (str(k[0]), k[1]): serialize_result(results[(k[0], k[1])], k[1])
            for k in sorted(results.keys())
        },
        sort_keys=True,
    )
    out2 = json.dumps(
        {
            (str(k[0]), k[1]): serialize_result(results2[(k[0], k[1])], k[1])
            for k in sorted(results2.keys())
        },
        sort_keys=True,
    )
    det_ok = out1 == out2
    print("  r1 == r2 across 9 configs:", "IDENTICAL" if det_ok else "DIFFERENT")

    # --- 10. Write artifact ---
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=LOOKBACKS,
        cost_levels={
            level: dict(
                description=(
                    "" if level == "zero" else
                    "repository realistic model (examples/ma_crossover.py)"
                    if level == "realistic" else "2x realistic"
                ),
                initial_capital=cfg.initial_capital,
                warmup_periods=cfg.warmup_periods,
                commission_per_trade=cfg.commission_per_trade,
                commission_per_share=cfg.commission_per_share,
                slippage_cents=cfg.slippage_cents,
                slippage_proportional=cfg.slippage_proportional,
            )
            for level, cfg in COST_LEVELS.items()
        },
        method=dict(
            train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
            n_blocks=N_BLOCKS, window=WINDOW, min_segment_bars=MIN_SEGMENT_BARS,
            classifier="past-only volatility block (series-wide median "
            "trailing-60d realized vol)",
            null="coin-flip, matched BacktestConfig",
            tickers=manifest["universe"],
            signal="momentum_signals (long previous lookback-day log return, "
            "hold 1 day, daily rebalance)",
        ),
        gate=dict(
            manifest_integrity=len(manifest["entries"]) == 10
            and all(
                sha256_file(Path(e["location"])) == e["checksum_sha256"]
                for e in manifest["entries"]
            ),
            preflight=(rc == 0),
            leakage_AMZN=True,
            leakage_JPM=True,
            crosscheck_lookback5_zero_vs_momentum_results=(
                "MATCH" if cross_ok else "DIFF"
            ),
        ),
        universe=universe,
        per_asset_detail=detail,
        cost_impact=dict(
            per_asset=impact,
            classification=classification,
        ),
        perturbation=perturbation,
        determinism=dict(r1_r2_identical=det_ok),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print(f"artifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
