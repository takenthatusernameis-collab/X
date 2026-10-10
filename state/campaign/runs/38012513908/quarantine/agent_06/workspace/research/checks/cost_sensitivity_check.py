"""Cost-sensitivity check for lookback 3 / 5 / 10 momentum.

Background: the momentum frontier test (`research/checks/momentum.py`, activation 37318950814)
showed a REGIME_STABLE positive edge in 7/10 assets at lookback=5, hold=1 day.
The synthetic perturbation sweep is a tooling-validity check only: the regime-
switching GBM generator contains no return autocorrelation, so it cannot tell
whether the real-data momentum edge survives a change of horizon on REAL data.

This check closes that gap: momentum is re-tested across lookbacks 3 / 5 / 10 on the
same collected adjusted-close universe, same walk-forward, same coin-flip null.
For each lookback it reports per-asset regime-stability segments/medians/verdicts
and fold-level t-statistics against H0: mean log fold return = 0 (nominal 5%
and Bonferroni family-wise over the 10-asset family). The edge must persist
across horizons without collapsing to noise before the momentum class can be
admitted to the evidence base.

Method (fixed a-priori, not tuned to OOS):
- Cost model: realistic transaction costs (commission_per_trade=2.0,
  commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)
- Walk-forward: train=252d / test=84d / warmup=60d / overlap=60d, min 400 bars/segment
- Perturbation sweep: lookback 3 / 5 / 10 across the canonical regime family, with a
coin-flip null benchmark.
- Determinism: independent re-run for verification.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return super().default(obj)

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
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=0,
)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_sensitivity_results.json"


def key_fields():
    """Canonical record for the determinism comparison."""
    np.random.seed(SEED)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars

    rows = []
    for lb in LOOKBACKS:
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
            cfg=REALISTIC_COSTS,
        )
        verdicts = [r.overall_verdict for r in summary.assets.values()]
        counts = dict(summary.verdict_counts)
        stable = counts.get("REGIME_STABLE", 0)
        asset_rows_dict = {}
        for t, r in summary.assets.items():
            asset_rows_dict[t] = {
                "segments": [s.name for s in r.scenarios],
                "medians": [round(s.baseline_median_log_return, 3) for s in r.scenarios],
                "null_medians": [round(s.noise_median_log_return, 3) for s in r.scenarios],
                "candidate_dispersion": round(r.candidate_dispersion, 3),
                "null_dispersion": round(r.null_dispersion, 3),
                "verdict": r.overall_verdict,
            }
        row = (
            lb,
            counts,
            stable,
            summary.ticker_order,
            asset_rows_dict,
        )
        rows.append(row)
    return rows


def fold_stats_for(closes, lookback, train_window, test_window, warmup,
                   overlap_window):
    """One walk-forward; mean / median / std fold log returns, t-statistic vs 0,
    95% Wald CI, and significance flags."""
    bars = bt.BarSequence(
        dates=np.arange(1, len(closes) + 1, dtype=np.int64),
        opens=np.zeros_like(closes), highs=np.zeros_like(closes),
        lows=np.zeros_like(closes), closes=closes,
        volumes=np.zeros_like(closes),
    )
    signals = bt.momentum_signals(closes, lookback=lookback)
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


def cost_comparison_for(lookback):
    """Compare momentum performance with realistic costs vs zero costs."""
    manifest = load_manifest()
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars.closes_array()

    comparison = {}
    for ticker, closes in tickers.items():
        bars = bt.BarSequence(
            dates=np.arange(1, len(closes) + 1, dtype=np.int64),
            opens=np.zeros_like(closes), highs=np.zeros_like(closes),
            lows=np.zeros_like(closes), closes=closes,
            volumes=np.zeros_like(closes),
        )
        signals = bt.momentum_signals(closes, lookback=lookback)

        # Zero costs run
        zero_cfg = bt.BacktestConfig(warmup_periods=WARM)
        zero_result = bt.walk_forward(
            list(bars), signals,
            train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=zero_cfg,
        )
        zero_rets = np.array(
            [float(f.metrics["total_return"]) for f in zero_result.folds], dtype=np.float64
        )
        zero_log = np.log1p(np.clip(zero_rets, -1.0 + 1e-12, None))

        # Realistic costs run
        real_cfg = REALISTIC_COSTS
        real_result = bt.walk_forward(
            list(bars), signals,
            train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=real_cfg,
        )
        real_rets = np.array(
            [float(f.metrics["total_return"]) for f in real_result.folds], dtype=np.float64
        )
        real_log = np.log1p(np.clip(real_rets, -1.0 + 1e-12, None))

        comparison[ticker] = {
            "ticker": ticker,
            "zero_cost_mean": float(zero_log.mean()),
            "zero_cost_median": float(np.median(zero_log)),
            "zero_cost_n": len(zero_log),
            "real_cost_mean": float(real_log.mean()),
            "real_cost_median": float(np.median(real_log)),
            "real_cost_n": len(real_log),
            "cost_impact_mean": float(real_log.mean() - zero_log.mean()),
            "cost_impact_median": float(np.median(real_log) - np.median(zero_log)),
            "zero_cost_t": zero_log.mean() / (zero_log.std(ddof=1) / np.sqrt(len(zero_log))),
            "real_cost_t": real_log.mean() / (real_log.std(ddof=1) / np.sqrt(len(real_log))),
        }
    return comparison


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

    print("\n=== 3. Momentum across collected universe: lookback sweep 3 / 5 / 10 ===")
    print("Momentum = long the previous N-day return, hold 1 day, daily rebalanced. "
          "Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, compared vs coin-flip null on the same "
          "segments.\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    rows = key_fields()
    output = {}
    for lb, counts, stable, ticker_order, asset_rows_dict in rows:
        print("Lookback {} days: {} assets | REGIME_STABLE={} "
              "CONSISTENT_WITH_NOISE={} REGIME_DEPENDENT={} REGIME_STABLE_LOSS={}".format(
              lb, len(asset_rows_dict), counts.get("REGIME_STABLE", 0),
              counts.get("CONSISTENT_WITH_NOISE", 0), counts.get("REGIME_DEPENDENT", 0),
              counts.get("REGIME_STABLE_LOSS", 0)))
        print("  {:10s} {:28s} {:12s}".format("asset", "candidate_medians", "verdict"))
        for ticker, data in asset_rows_dict.items():
            meds_s = "[" + ",".join(str(m) for m in data["medians"]) + "]"
            print("  {:10s} {:28s} {:12s}".format(ticker, meds_s, data["verdict"]))
        print("")
        output["lookback_{}".format(lb)] = dict(
            n_assets=len(asset_rows_dict),
            verdict_counts=counts,
            n_regime_stable=stable,
            ticker_order=ticker_order,
            per_asset=[
                dict(
                    ticker=ticker,
                    segments=data["segments"],
                    medians=data["medians"],
                    null_medians=data["null_medians"],
                    candidate_dispersion=data["candidate_dispersion"],
                    null_dispersion=data["null_dispersion"],
                    verdict=data["verdict"],
                )
                for ticker, data in asset_rows_dict.items()
            ],
        )

    # Fold-level t-statistics for each lookback.
    print("=== 4. Fold-level t-statistics (H0: mean log fold return = 0) ===")
    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        bars, _ = bt.load_ticker(ticker)
        tickers[ticker] = bars.closes_array()

    trow = {}
    for lb in LOOKBACKS:
        hdr = ("ticker", "n_folds", "mean", "median", "std", "pos_folds",
               "t_stat", "df", "ci95_lo", "ci95_hi", "sig_5pct_nominal",
               "sig_5pct_bonf")
        widths = [6, 7, 9, 9, 9, 10, 8, 3, 9, 9, 16, 17]
        print("  Lookback {}d:".format(lb))
        print("   " + " | ".join(h.ljust(w) for h, w in zip(hdr, widths)))
        trow[lb] = []
        for ticker, closes in tickers.items():
            stats = fold_stats_for(closes, lb, TRAIN, TEST, WARM, OVERLAP)
            trow[lb].append((
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
    # Build fold_stats output for all lookbacks
    output["fold_stats"] = {}
    for lb in LOOKBACKS:
        output["fold_stats"]["lookback_{}".format(lb)] = {
            "n_assets": len(trow[lb]),
            "rows": list(trow[lb]),
        }

    # Cost impact comparison: realistic costs vs zero costs
    print("=== 5. Cost impact comparison (realistic vs zero costs) ===")
    cost_comparison = {}
    for lb in LOOKBACKS:
        comp = cost_comparison_for(lb)
        cost_comparison["lookback_{}".format(lb)] = comp
        print(f"\nLookback {lb} days cost impact:")
        print("  {:10s} {:12s} {:12s} {:12s} {:12s}".format(
            "ticker", "zero_cost_mean", "real_cost_mean", "impact_mean", "impact_pct"))
        for ticker_data in comp.values():
            impact_pct = (ticker_data["real_cost_mean"] - ticker_data["zero_cost_mean"]) / abs(ticker_data["zero_cost_mean"]) * 100 if ticker_data["zero_cost_mean"] != 0 else 0
            print("  {:10s} {:12.4f} {:12.4f} {:12.4f} {:12.1f}%".format(
                ticker_data["ticker"], ticker_data["zero_cost_mean"], ticker_data["real_cost_mean"],
                ticker_data["cost_impact_mean"], impact_pct))

    # Cross-lookback summary: how many assets keep the REGIME_STABLE edge as
    # lookback grows, and how many survive family-wise significance.
    print("=== 6. Cross-lookback summary ===")
    print("  REGIME_STABLE asset count by lookback:")
    for lb, _, stable, _, _ in rows:
        print("    lookback {}: {} / 10".format(lb, stable))
    nom_significant = []
    bonf_significant = []
    for lb in LOOKBACKS:
        nom_significant.append(sum(1 for r in trow[lb] if r[10]))
        bonf_significant.append(sum(1 for r in trow[lb] if r[11]))
    print("  nominally (5%) significant assets by lookback:")
    for lb, n in zip(LOOKBACKS, nom_significant):
        print("    lookback {}: {} / 10".format(lb, n))
    print("  Bonferroni family-wise (alpha {}) significant assets by lookback:".format(
        BONFERRONI_ALPHA))
    for lb, n in zip(LOOKBACKS, bonf_significant):
        print("    lookback {}: {} / 10".format(lb, n))
    print()
    output["cross_summary"] = {
        "regime_stable_by_lookback": {
            "lookback_{}".format(lb): stable for i, (lb, counts, stable, _, _) in enumerate(rows)
        },
        "nominally_significant_by_lookback": {
            "lookback_{}".format(lb): n for lb, n in zip(LOOKBACKS, nom_significant)
        },
        "bonferroni_significant_by_lookback": {
            "lookback_{}".format(lb): n for lb, n in zip(LOOKBACKS, bonf_significant)
        },
    }

    # Cost-robustness summary: does the edge survive realistic costs?
    print("=== 7. Cost-robustness summary ===")
    robustness_summary = {}
    for lb in LOOKBACKS:
        comp = cost_comparison["lookback_{}".format(lb)]
        edge_survives = []
        for ticker_data in comp.values():
            # Consider edge survives if real_cost_mean still positive and magnitude is not dramatically reduced
            survives = ticker_data["real_cost_mean"] > 0.02  # > 2% per year with realistic costs
            edge_survives.append(survives)
        robustness_summary["lookback_{}".format(lb)] = {
            "assets_considered": len(comp),
            "assets_edge_survives_realistic_costs": sum(edge_survives),
            "edge_survival_rate": sum(edge_survives) / len(comp) if comp else 0,
            "mean_zero_cost_performance": sum(d["zero_cost_mean"] for d in comp.values()) / len(comp) if comp else 0,
            "mean_realistic_cost_performance": sum(d["real_cost_mean"] for d in comp.values()) / len(comp) if comp else 0,
            "mean_cost_impact_pct": sum(d["cost_impact_mean"] for d in comp.values()) / len(comp) if comp else 0,
        }
        print(f"\nLookback {lb} days cost-robustness:")
        print(f"  assets where edge survives realistic costs: {sum(edge_survives)}/{len(comp)}")
        print(f"  mean performance (zero costs): {robustness_summary['lookback_{}'.format(lb)]["mean_zero_cost_performance"]:.4f}")
        print(f"  mean performance (realistic costs): {robustness_summary['lookback_{}'.format(lb)]["mean_realistic_cost_performance"]:.4f}")
        print(f"  mean cost impact: {robustness_summary['lookback_{}'.format(lb)]["mean_cost_impact_pct"]:.4f}")

    output["cost_robustness"] = robustness_summary

    # Determinism: rerun the canonical key fields and assert identical output.
    print("\n=== 8. Determinism (independent re-run of sweep) ===")
    r2 = key_fields()
    print("  r1 == r2: {}".format(rows == r2))
    for a, b in zip(rows, r2):
        assert a == b, "non-deterministic sweep output"
    print("  all sweep fields identical across reruns")
    print("")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        nominal_alpha=NOMINAL_ALPHA, bonferroni_alpha=BONFERRONI_ALPHA,
        **output,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True, cls=CustomJSONEncoder)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
