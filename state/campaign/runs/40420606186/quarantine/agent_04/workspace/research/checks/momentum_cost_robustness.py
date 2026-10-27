"""Cost sensitivity check for short-horizon momentum (lookback 3/5/10).

Background: The momentum signal class shows REGIME_STABLE positive edge in
7/10 assets at lookback=5 on the collected universe, but execution-cost
robustness is not yet validated. This check subjects the same edge to
realistic transaction costs and asks: does the economic value survive?

Method (fixed a-priori, not tuned to OOS):
- Long the previous N-day return, hold 1 day, daily rebalance.
- Use realistic cost model (commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005).
- Walk-forward per volatility regime (4 blocks, window=60, min 400 bars/segment).
- Compare vs coin-flip null on the same segments.
- Focus on lookback 3, 5, 10.

Deliverable: artifact with cost-robustness verdict per lookback.
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
NOMINAL_ALPHA = 0.05
BONFERRONI_ALPHA = NOMINAL_ALPHA / 10
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"

REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=False,
)

def key_fields_for_lookback(lookback):
    """Canonical record for determinism comparison."""
    np.random.seed(SEED)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars

    rows = []
    for lb in [lookback]:
        summary = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lb),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS,
                                                           window=WINDOW),
            baseline=(("lookback", lb),),
            param_grid=[{"lookback": lb}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        verdicts = [r.overall_verdict for r in summary.assets.values()]
        counts = dict(summary.verdict_counts)
        stable = counts.get("REGIME_STABLE", 0)
        row = (
            lb,
            counts,
            stable,
            summary.ticker_order,
            [
                (
                    t,
                    [s.name for s in r.scenarios],
                    [round(s.baseline_median_log_return, 3) for s in r.scenarios],
                    [round(s.noise_median_log_return, 3) for s in r.scenarios],
                    round(r.candidate_dispersion, 3),
                    round(r.null_dispersion, 3),
                    r.overall_verdict,
                )
                for t, r in summary.assets.items()
            ],
        )
        rows.append(row)
    return rows, summary

def run_cost_robustness():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Momentum Cost-Robustness Check (lookback 3/5/10) ===")
    print("Signal: long previous N-day return, hold 1 day, daily rebalance.")
    print(f"Realistic costs: commission_per_trade={REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share={REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    print(f"Universe: {len(manifest['universe'])} assets, {len(manifest['entries'])} entries")
    print(f"Params: walk-forward train={TRAIN}d / test={TEST}d / warmup={WARM}d / "
          f"overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment")

    results = {}

    for lb in LOOKBACKS:
        print(f"\n--- Lookback {lb} days ---")

        rows, summary = key_fields_for_lookback(lb)

        print(f"  Universe-wide verdict counts: {summary.verdict_counts}")
        stable = summary.verdict_counts.get("REGIME_STABLE", 0)
        print(f"  REGIME_STABLE assets: {stable} / {len(summary.assets)}")

        # Per-asset breakdown
        print("  Per-asset results:")
        print("    {:10s} {:24s} {:24s} {:>10s}  {:>10s}  {:8s}".format(
            "asset", "median_log_ret", "noise_median", "cand_disp", "null_disp", "verdict"))
        for ticker, r in summary.assets.items():
            med_list = [round(s.baseline_median_log_return, 3) for s in r.scenarios]
            null_list = [round(s.noise_median_log_return, 3) for s in r.scenarios]
            med_str = "[" + ",".join(f"{m:+.3f}" for m in med_list) + "]"
            null_str = "[" + ",".join(f"{m:+.3f}" for m in null_list) + "]"
            cand_disp_str = f"{r.candidate_dispersion:+10.3f}"
            null_disp_str = f"{r.null_dispersion:+10.3f}"
            print(f"    {ticker:10s} {med_str:24s} {null_str:24s} {cand_disp_str}  {null_disp_str}  {r.overall_verdict:8s}")

        # Record results for artifact
        results[f"lookback_{lb}"] = {
            "verdict_counts": summary.verdict_counts,
            "n_regime_stable": stable,
            "ticker_order": summary.ticker_order,
            "per_asset": {
                ticker: {
                    "segments": [s.name for s in r.scenarios],
                    "medians": [round(s.baseline_median_log_return, 3) for s in r.scenarios],
                    "null_medians": [round(s.noise_median_log_return, 3) for s in r.scenarios],
                    "candidate_dispersion": round(r.candidate_dispersion, 3),
                    "null_dispersion": round(r.null_dispersion, 3),
                    "verdict": r.overall_verdict,
                    "n_folds": r.n_folds,
                }
                for ticker, r in summary.assets.items()
            },
        }

    # Determinism: rerun key fields and assert identical output
    print("\n=== Determinism check ===")
    all_deterministic = True
    for lb in LOOKBACKS:
        rows1, _ = key_fields_for_lookback(lb)
        rows2, _ = key_fields_for_lookback(lb)
        deterministic = rows1 == rows2
        print(f"  lookback {lb}: deterministic = {deterministic}")
        if not deterministic:
            all_deterministic = False
    assert all_deterministic, "non-deterministic output"

    # Write artifact
    artifact = {
        "dataset_id": DATASET_ID,
        "seed": SEED,
        "lookbacks": list(LOOKBACKS),
        "train": TRAIN, "test": TEST, "warmup": WARM, "overlap": OVERLAP,
        "window": WINDOW, "n_blocks": N_BLOCKS, "min_segment_bars": MIN_SEGMENT_BARS,
        "nominal_alpha": NOMINAL_ALPHA, "bonferroni_alpha": BONFERRONI_ALPHA,
        "cost_config": {
            "commission_per_trade": REALISTIC_COSTS.commission_per_trade,
            "commission_per_share": REALISTIC_COSTS.commission_per_share,
            "slippage_cents": REALISTIC_COSTS.slippage_cents,
            "slippage_proportional": REALISTIC_COSTS.slippage_proportional,
        },
        **results,
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nArtifact written to: {ARTIFACT_PATH}")
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-robustness check: lookback 3/5/10 on collected universe.")

    return artifact

if __name__ == "__main__":
    sys.exit(0 if run_cost_robustness() else 1)