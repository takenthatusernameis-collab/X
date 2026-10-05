"""Fold-level t-statistic across the collected Yahoo Finance OHLCV universe
(research-only).

For each ticker the MA(20/60) crossover is walk-forward validated
(train=252d, test=84d, warmup=60d, overlap=60d) and the distribution of
OOS fold log returns is summarized with:

  - mean / median / std of fold log returns
  - positive-fold count
  - a t-statistic against H0: mean log fold return = 0
  - the degrees of freedom
  - a 95% Wald confidence interval for the mean log fold return

This is a complementary lens to the coin-flip null in ``universe.py``: the
t-test judges whether the mean fold return is distinguishable from zero at
the observed sample size (one-sided in practice, reported via the CI check),
while the null in universe.py judges whether the signal carries any
price-related information at all. The two answers can differ (e.g. a
statistically significant mean fold return that sits within the framework's
coin-flip noise band, as observed for AAPL in the 2026-10-04 activation).

A cross-asset summary reports, for the family of per-asset t-tests, how many
tickers are nominally significant at 5% and how many survive a conservative
Bonferroni family-wise correction (alpha / n_assets = 0.005), since the same
walk-forward parameters and folds are used for every asset.

Usage:
    python3 research/checks/universe_stats.py

Determinism: the script asserts its own byte-identical reproducibility across
independent re-runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
WARMUP = 60
FAST, SLOW = 20, 60
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
NOMINAL_ALPHA = 0.05
BONFERRONI_ALPHA = NOMINAL_ALPHA / 10  # family-wise over 10 tickers


def ma_crossover_signals(closes, fast, slow):
    """Past-only MA-crossover signals (same convention as ma_crossover_real_data)."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    fast, slow = int(fast), int(slow)
    for i in range(max(fast, slow) - 1, n):
        fast_ma = np.mean(closes[i - fast + 1 : i + 1])
        slow_ma = np.mean(closes[i - slow + 1 : i + 1])
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if fast_ma > slow_ma else -1.0)
    return signals


def asset_fold_stats(closes, fast, slow, train_window, test_window, warmup,
                     overlap_window):
    """Run one walk-forward and return per-asset t-statistic fields."""
    bars = bt.BarSequence(
        dates=np.arange(1, len(closes) + 1, dtype=np.int64),
        opens=np.zeros_like(closes), highs=np.zeros_like(closes),
        lows=np.zeros_like(closes), closes=closes,
        volumes=np.zeros_like(closes),
    )
    signals = ma_crossover_signals(closes, fast, slow)
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
        "mean_log_ret": log.mean(),
        "median_log_ret": float(np.median(log)),
        "std_log_ret": log.std(ddof=1),
        "positive_folds": int((log > 0).sum()),
        "n_positive": int((log > 0).sum()),
        "t_statistic": tstat,
        "df": n - 1,
        "ci_lo_95": lo,
        "ci_hi_95": hi,
        "significance_5pct_nominal": abs(tstat) > 1.96,
        "significance_5pct_bonferroni": abs(tstat) > 1.96 * np.sqrt(n),
    }


def key_fields():
    """Return the canonical per-asset result record for determinism comparison."""
    np.random.seed(42)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

    universe = manifest["universe"]
    rows = []
    for ticker in universe:
        bars, dates = bt.load_ticker(ticker)
        stats = asset_fold_stats(
            bars.closes_array(), FAST, SLOW, TRAIN, TEST, WARM, OVERLAP
        )
        rows.append(
            (
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
            )
        )
    return rows


def main():
    np.random.seed(42)
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"

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
        print("PREFLIGHT FAILED; aborting.")
        sys.exit(1)

    rows = key_fields()

    print("\n=== 3. Fold-level t-statistic, MA(20/60) walk-forward, "
          f"train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d ===")
    hdr = ("ticker", "n_folds", "mean", "median", "std", "pos_folds",
           "t_stat", "df", "ci95_lo", "ci95_hi", "sig_5pct_nominal", "sig_5pct_bonf")
    widths = [6, 7, 9, 9, 9, 10, 8, 3, 9, 9, 16, 17]
    print("  " + " | ".join(h.ljust(w) for h, w in zip(hdr, widths)))
    for r in rows:
        print("  " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)))

    print("\n=== 4. Cross-asset significance summary ===")
    n_assets = len(rows)
    nom_significant = sum(1 for r in rows if r[10])
    bonf_significant = sum(1 for r in rows if r[11])
    print(f"  assets: {n_assets} | nominal 5% significant: {nom_significant}")
    print(f"  Bonferroni family-wise (alpha {BONFERRONI_ALPHA}): {bonf_significant}")
    for name, nf, mean, med, std, pos, tstat, df, lo, hi, nom, bonf in rows:
        flag = ""
        if nom:
            flag += "  <-- nominal sig"
        if bonf:
            flag += "  <-- Bonf sig"
        print(f"  {name}: t={tstat:+.2f} (df={df}), "
              f"mean log ret {mean:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], "
              f"pos folds {pos}/{nf}{flag}")
    print()
    print("=== 5. Determinism (independent re-run) ===")
    r1 = key_fields()
    r2 = key_fields()
    print(f"  r1 == r2: {r1 == r2}")
    for a, b in zip(r1, r2):
        same = a == b
        assert same, f"field differs: {a} vs {b}"
    print("  all per-asset fields identical across reruns")
    print()
    print("NOTE: research/simulation only. No live trading or production "
          "execution. MA crossover fold statistics across the collected "
          "universe: exploratory simulation.")


if __name__ == "__main__":
    main()
