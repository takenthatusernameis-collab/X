"""Momentum robustness: walk-forward on the collected universe across holding periods.

Test whether lookback-5 momentum survives a small predeclared holding-period grid
without becoming a single-point timing artifact.

This test extends the momentum frontier by examining holding-period sensitivity.
While lookback=5 is fixed (strongest evidence), we vary the holding period
to test temporal robustness.

Method (fixed a-priori, not tuned to OOS): long the previous 5-day return;
hold for H days (H from predeclared grid), daily rebalance. Walk-forward
per segment train=252d/test=84d/warmup=60d/overlap=60d; min 400 bars/segment.
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
HOLDING_PERIODS = (1, 3, 5, 7)  # Predeclared holding-period grid
NOMINAL_ALPHA = 0.05
BONFERRONI_ALPHA = NOMINAL_ALPHA / 10
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_holding_period_results.json"


def key_fields_for_holding_period(holding_period):
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
    for hp in HOLDING_PERIODS:
        summary = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: bt.momentum_signals_fixed_hold(c, lookback=5, holding_period=hp),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", 5), ("holding_period", hp)),
            param_grid=[{"lookback": 5, "holding_period": hp}],
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
            hp,
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
    return rows


def fold_stats_for_holding_period(closes, holding_period, train_window, test_window, warmup, overlap_window):
    """One walk-forward for a specific holding period."""
    bars = bt.BarSequence(
        dates=np.arange(1, len(closes) + 1, dtype=np.int64),
        opens=np.zeros_like(closes), highs=np.zeros_like(closes),
        lows=np.zeros_like(closes), closes=closes,
        volumes=np.zeros_like(closes),
    )
    signals = bt.momentum_signals_fixed_hold(closes, lookback=5, holding_period=holding_period)
    cfg = bt.BacktestConfig(warmup_periods=warmup)
    result = bt.walk_forward(
        list(bars), signals,
        train_window=train_window, test_window=test_window,
        warmup=warmup, overlap_window=overlap_window, cfg=cfg,
    )
    rets = np.array(
        [float(f.metrics["total_return"]) for f in result.folds], dtype=np.float64
    )
    log = np.log1p(np.clip(rets, -1.0 + 1e-12, None))
    n = len(log)
    se = log.std(ddof=1) / np.sqrt(n)
    tstat = log.mean() / se
    hi = log.mean() + 1.96 * se
    lo = log.mean() - 1.96 * se
    return {
        "n_folds": n,
        "mean_log_ret": float(log.mean()),
        "median_log_ret": float(np.median(log)),
        "std_log_ret": float(log.std(ddof=1)),
        "positive_folds": int((log > 0).sum()),
        "t_statistic": tstat,
        "df": n - 1,
        "ci_lo_95": lo,
        "ci_hi_95": hi,
        "significance_5pct_nominal": abs(tstat) > 1.96,
        "significance_5pct_bonferroni": abs(tstat) > 1.96 * np.sqrt(n),
    }


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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Momentum lookback=5 across collected universe: holding-period sweep {} days ===".format(HOLDING_PERIODS))
    print("Momentum = long the previous 5-day return, hold H days, daily rebalanced. "
          "Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, compared vs coin-flip null on the same "
          "segments.\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    rows = key_fields_for_holding_period(1)
    output = {}
    for idx, (hp, counts, stable, ticker_order, asset_rows) in enumerate(rows):
        print("Holding period {} days: {} assets | REGIME_STABLE={} "
              "CONSISTENT_WITH_NOISE={} REGIME_DEPENDENT={} REGIME_STABLE_LOSS={}".format(
              hp, len(asset_rows), counts.get("REGIME_STABLE", 0),
              counts.get("CONSISTENT_WITH_NOISE", 0), counts.get("REGIME_DEPENDENT", 0),
              counts.get("REGIME_STABLE_LOSS", 0)))
        print("  {:10s} {:28s} {:12s}".format("asset", "candidate_medians", "verdict"))
        for asset_row in asset_rows:
            # asset_row is (ticker, segments, candidate_medians, null_medians, candidate_dispersion, null_dispersion, verdict)
            ticker = asset_row[0]
            segments = asset_row[1]
            candidate_medians = asset_row[2]
            null_medians = asset_row[3]
            candidate_dispersion = asset_row[4]
            null_dispersion = asset_row[5]
            verdict = asset_row[6]
            meds_s = "[" + ",".join(str(m) for m in candidate_medians) + "]"
            print("  {:10s} {:28s} {:12s}".format(ticker, meds_s, verdict))
        print("")
        output["holding_period_{}".format(hp)] = dict(
            n_assets=len(asset_rows),
            verdict_counts=counts,
            n_regime_stable=stable,
            ticker_order=ticker_order,
            per_asset=[
                dict(
                    ticker=asset_row[0],
                    segments=asset_row[1],
                    medians=asset_row[2],
                    null_medians=asset_row[3],
                    candidate_dispersion=asset_row[4],
                    null_dispersion=asset_row[5],
                    verdict=asset_row[6],
                )
                for asset_row in asset_rows
            ],
        )

    # Fold-level t-statistics for each holding period.
    print("=== 4. Fold-level t-statistics (H0: mean log fold return = 0) ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars.closes_array()

    trow = {}
    for hp in HOLDING_PERIODS:
        hdr = ("ticker", "n_folds", "mean", "median", "std", "pos_folds",
               "t_stat", "df", "ci95_lo", "ci95_hi", "sig_5pct_nominal",
               "sig_5pct_bonferroni")
        widths = [6, 7, 9, 9, 9, 10, 8, 3, 9, 9, 16, 17]
        print("  Holding period {}d:".format(hp))
        print("   " + " | ".join(h.ljust(w) for h, w in zip(hdr, widths)))
        trow[hp] = []
        for ticker, closes in tickers.items():
            stats = fold_stats_for_holding_period(closes, hp, TRAIN, TEST, WARM, OVERLAP)
            trow[hp].append((
                ticker,
                stats["n_folds"],
                round(stats["mean_log_ret"], 6),
                round(stats["median_log_ret"], 6),
                round(stats["std_log_ret"], 6),
                stats["positive_folds"],
                round(stats["t_statistic"], 4),
                stats["df"],
                round(stats["ci_lo_95"], 6),
                round(stats["ci_hi_95"], 6),
                stats["significance_5pct_nominal"],
                stats["significance_5pct_bonferroni"],
            ))
            print("   " + " | ".join(str(v).ljust(w) for v, w in zip(
                (ticker, stats["n_folds"], round(stats["mean_log_ret"], 6),
                 round(stats["median_log_ret"], 6), round(stats["std_log_ret"], 6),
                 stats["positive_folds"], round(stats["t_statistic"], 4),
                 stats["df"], round(stats["ci_lo_95"], 6), round(stats["ci_hi_95"], 6),
                 stats["significance_5pct_nominal"],
                 stats["significance_5pct_bonferroni"]), widths)))
        print()

    # Cross-holding-period summary: how many assets keep the REGIME_STABLE edge as
    # holding period grows.
    print("=== 5. Cross-holding-period summary ===")
    print("  REGIME_STABLE asset count by holding period:")
    for hp, _, stable, _, _ in rows:
        print("    holding period {}: {} / 10".format(hp, stable))
    nom_significant = []
    bonf_significant = []
    for hp in HOLDING_PERIODS:
        nom_significant.append(sum(1 for r in trow[hp] if r[10]))
        bonf_significant.append(sum(1 for r in trow[hp] if r[11]))
    print("  nominally (5%) significant assets by holding period:")
    for hp, n in zip(HOLDING_PERIODS, nom_significant):
        print("    holding period {}: {} / 10".format(hp, n))
    print("  Bonferroni family-wise (alpha {}) significant assets by holding period:".format(
        BONFERRONI_ALPHA))
    for hp, n in zip(HOLDING_PERIODS, bonf_significant):
        print("    holding period {}: {} / 10".format(hp, n))
    print()
    output["fold_stats"] = {
        "holding_period_{}".format(hp): {
            "n_assets": len(trow[hp]),
            "rows": list(trow[hp]),
        } for hp in HOLDING_PERIODS
    }
    output["cross_summary"] = {
        "regime_stable_by_holding_period": {
            "holding_period_{}".format(hp): dict(rows[i][1]) for i, hp in enumerate(HOLDING_PERIODS)
        },
        "nominally_significant_by_holding_period": {
            "holding_period_{}".format(hp): n for hp, n in zip(HOLDING_PERIODS, nom_significant)
        },
        "bonferroni_significant_by_holding_period": {
            "holding_period_{}".format(hp): n for hp, n in zip(HOLDING_PERIODS, bonf_significant)
        },
    }

    # Determinism: rerun the canonical key fields and assert identical output.
    r2 = key_fields_for_holding_period(1)
    print("=== 6. Determinism (independent re-run of holding-period sweep) ===")
    print("  r1 == r2: {}".format(rows == r2))
    for a, b in zip(rows, r2):
        assert a == b, "non-deterministic holding-period sweep output"
    print("  all sweep fields identical across reruns")
    print("")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=5,
        holding_periods=list(HOLDING_PERIODS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        nominal_alpha=NOMINAL_ALPHA, bonferroni_alpha=BONFERRONI_ALPHA,
        **output,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback=5 holding-period sweep on the collected "
          "universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())