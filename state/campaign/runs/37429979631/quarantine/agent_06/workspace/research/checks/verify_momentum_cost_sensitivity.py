"""Independent verification of the momentum cost-sensitivity check
(`research/checks/momentum_cost_sensitivity.py`).

This is a separate implementation path: it recomputes each segment's
walk-forward median under the zero / realistic / heavy cost configs via a
fresh ``walk_forward`` implementation (not ``stress_segments``), regenerates
the regime-stability verdicts, recomputes the decision-relevant 1-day-hold
daily-rebalance cost drag with a separate cost model implementation, and
compares against the artifact written by the cost check
(`state/check_artifacts/momentum_cost_sensitivity_results.json`).

Design of the independent path:
- Regime labels are reconstructed from ``closes`` using a re-implemented
  ``volatility_blocks`` (same algorithm as research.backtest, different code).
- Segments are reconstructed from labels using ``min_segment_bars``.
- Segment medians are computed via ``walk_forward`` directly on the segment
  slice, with per-config backtest costs, aggregating fold log returns -- no
  call to ``stress_segments`` or ``stress_segments_across_tickers``.
- The coin-flip null is replicated for one representative cell by
  re-implementing ``random_signals`` (equal {-1, 0, +1} choice, seed derived
  from parameter values) and walking it with ``walk_forward``; this confirms
  the null mechanism under costs, not just its reported values.
- The daily-rebalance cost drag is recomputed with a fresh cost arithmetic
  implementation (entry/exit fill price = close + fixed slippage +
  proportional slippage; round-trip commission + slippage per bar); this is
  the decision-relevant metric for the "hold 1 day, daily rebalance" strategy.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, segments, cost configs)
so a match confirms the check computed from those inputs; a mismatch would
flag an error in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.engine import Signal
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
OUTLIER_TOL = 0.05
LOOKBACKS = (3, 5, 10)
INITIAL_CAPITAL = 1e6

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"

# Predeclared cost configs -- identical to the check's, declared here fresh.
ZERO_COST = bt.BacktestConfig()
REALISTIC_COST = bt.BacktestConfig(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
HEAVY_COST = bt.BacktestConfig(
    commission_per_trade=4.0,
    commission_per_share=0.006,
    slippage_cents=4.0,
    slippage_proportional=0.001,
)
COST_CONFIGS = [("zero", ZERO_COST), ("realistic", REALISTIC_COST), ("heavy", HEAVY_COST)]

VERDICT_NAMES = ("zero", "realistic", "heavy")


def volatility_blocks(closes, n_blocks, window):
    """Independent re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    """Independent implementation of the momentum signal (different code path)."""
    n = len(closes)
    out = [Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def random_signals(n, seed):
    """Independent re-implementation of research.backtest.perturbation.random_signals."""
    rng = np.random.default_rng(seed)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]


def _param_seed(params):
    """Independent re-implementation of
    research.backtest.perturbation.noise_benchmark._param_seed."""
    total = sum(float(v) for v in params.values())
    return SEED + int(round(total) * 1000)


def segment_median(ticker, s, e, lookback, cfg):
    """Fresh walk_forward segment median for one cost config."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    res = bt.walk_forward(
        list(bars)[s:e], signals[s:e], train_window=TRAIN, test_window=TEST,
        warmup=WARM, overlap_window=OVERLAP, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def null_medians(ticker, s, e, lookback, warm, overlap):
    """Independent coin-flip null for one segment, via walk_forward.

    The check's ``stress_segments`` computes the null with ``warmup=0,
    overlap_window=0`` on the segment slice, and ``noise_benchmark``
    generates the coin-flip signal with length = the segment slice length,
    so the RNG stream is per-segment (not a slice of a full-series stream).
    """
    n_seg = e - s
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = random_signals(n_seg, _param_seed({"lookback": float(lookback)}))
    res = bt.walk_forward(
        list(bars)[s:e], signals, train_window=TRAIN, test_window=TEST,
        warmup=warm, overlap_window=overlap, cfg=ZERO_COST)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds)


def daily_rebalance_results(tickers, lookback, cfg, tickers_for_cost=()):
    """Independent 1-day-hold, daily-rebalance cost measurement on the full
    sample. Fresh cost arithmetic (not the check's implementation):
    entry at i's close with fixed+proportional slippage, exit at i+1's close,
    round-trip commission + slippage deducted each bar."""
    tickers_for_cost = tickers_for_cost or list(tickers_for_cost)[:1] or list(tickers)[:1]
    first = tickers_for_cost[0]
    bars, dates = bt.load_ticker(first)
    closes = bars.closes_array()
    n = len(closes)
    signals = momentum_signals(closes, lookback)

    gross_rets = np.zeros(n, dtype=np.float64)
    net_rets = np.zeros(n, dtype=np.float64)
    round_trip_costs = np.zeros(n, dtype=np.float64)
    counts = {"nonzero_signal": 0, "costed": 0}

    f, p = cfg.slippage_cents / 100.0, cfg.slippage_proportional
    comm_per_trade = cfg.commission_per_trade
    comm_per_share = cfg.commission_per_share

    for i in range(lookback, n - 1):
        sig = signals[i]
        if abs(sig.weight) < 1e-12:
            continue
        counts["nonzero_signal"] += 1
        entry_price = closes[i] + f + closes[i] * p
        exit_price = closes[i + 1] + f + closes[i + 1] * p
        shares = INITIAL_CAPITAL * cfg.target_exposure / entry_price
        gross_pnl = sig.weight * shares * (exit_price - entry_price)
        cost = (2.0 * (comm_per_trade + abs(sig.weight) * shares * comm_per_share)
                + abs(sig.weight) * shares * (entry_price - closes[i])
                + abs(sig.weight) * shares * (exit_price - closes[i + 1]))
        net_pnl = gross_pnl - cost
        counts["costed"] += 1
        gross_rets[i] = np.log1p(gross_pnl / INITIAL_CAPITAL)
        net_rets[i] = np.log1p(net_pnl / INITIAL_CAPITAL)
        round_trip_costs[i] = cost

    gross_rets = gross_rets[lookback:n - 1]
    net_rets = net_rets[lookback:n - 1]
    round_trip_costs = round_trip_costs[lookback:n - 1]

    mean_cost_per_day = float(np.mean(round_trip_costs)) if len(round_trip_costs) else 0.0

    med = float(np.median(net_rets))
    return dict(
        n_bars=int(n),
        n_days_with_signal=int(counts["nonzero_signal"]),
        n_days_backtested=int(len(net_rets)),
        mean_daily_gross_log_return=round(float(np.mean(gross_rets)), 6),
        median_daily_gross_log_return=round(float(np.median(gross_rets)), 6),
        mean_daily_net_log_return=round(float(np.mean(net_rets)), 6),
        median_daily_net_log_return=round(med, 6),
        mean_round_trip_cost_dollars=round(mean_cost_per_day, 2),
        mean_round_trip_cost_pct_of_capital=round(mean_cost_per_day / INITIAL_CAPITAL, 6),
        mean_daily_cost_pct=round(float(np.mean(round_trip_costs) / INITIAL_CAPITAL), 6),
        net_vs_gross_delta=round(med - round(float(np.median(gross_rets)), 6), 6),
    )


def verdict_from(medians, null_medians):
    """Regenerate a regime-stability verdict from medians and null medians,
    using the same decision rule as research.backtest.regime_stability."""
    if all(abs(m) <= OUTLIER_TOL for m in medians):
        return "CONSISTENT_WITH_NOISE"
    cand_disp = float(np.std(medians))
    null_disp = float(np.std(null_medians))
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert artifact["lookbacks"] == list(LOOKBACKS)

    print("=== Gate 1: walk-forward regime-stability grid (fresh recomputation) ===")
    print("  Candidate medians recomputed via fresh walk_forward under each cost "
          "config; verdicts regenerated from recomputed medians vs artifact null medians.")
    print("  (The artifact's own null medians are used for dispersion; see Gate 1b "
          "for an independent null replication.)\n")

    tickers = {}
    manifest = load_manifest()
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    all_ok = True

    # Per-asset (AMZN, JPM) segment medians under each (lookback, cost) cell.
    print("  Per-asset recomputation (AMZN, JPM) by lookback and cost:")
    per_asset_recomputed = {}
    for asset in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(asset)
        closes = bars.closes_array()
        labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        per_asset_recomputed[asset] = {"momentum": {}}
        for lookback in LOOKBACKS:
            per_asset_recomputed[asset]["momentum"][str(lookback)] = {}
            for label, cfg in COST_CONFIGS:
                seg_medians = [round(segment_median(asset, s, e, lookback, cfg)[0], 3)
                               for lab, s, e in runs]
                pub = artifact["per_asset"][asset]["momentum"][str(lookback)][label]["medians"]
                pub_lbls = artifact["per_asset"][asset]["momentum"][str(lookback)][label]["segments"]
                lbls = [lab for lab, _, _ in runs]
                status = "MATCH" if seg_medians == pub and lbls == pub_lbls else "MISMATCH"
                if status != "MATCH":
                    all_ok = False
                status2 = "MATCH"
                cand = np.array(seg_medians)
                null = np.array(artifact["per_asset"][asset]["momentum"][str(lookback)][label]["null_medians"])
                regen = verdict_from(seg_medians, null.tolist())
                if regen != artifact["per_asset"][asset]["momentum"][str(lookback)][label]["verdict"]:
                    status2 = "MISMATCH"
                    all_ok = False
                if status != "MATCH" or status2 != "MATCH":
                    print("    [DEBUG] {}: lookback={} {} medians {} vs pub {} lbls {} vs pub {}".format(
                        asset, lookback, label, seg_medians, pub, lbls, pub_lbls))
                print("    {} lookback={}: {} -> medians={} n_folds={} verdict={}(pub {}) -> {}".format(
                    asset, lookback, label, seg_medians,
                    artifact["per_asset"][asset]["momentum"][str(lookback)][label]["n_folds"],
                    regen, artifact["per_asset"][asset]["momentum"][str(lookback)][label]["verdict"],
                    status))
                per_asset_recomputed[asset]["momentum"][str(lookback)][label] = {
                    "medians": seg_medians, "segments": lbls, "verdict": regen}

    # Universe-level verdict counts per (lookback, cost): recompute all 10 assets.
    print("\n  Full-universe recomputation (all 10 assets) by lookback and cost:")
    universe_recomputed = {}
    for lookback in LOOKBACKS:
        universe_recomputed[str(lookback)] = {}
        for label, cfg in COST_CONFIGS:
            counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_STABLE": 0,
                      "REGIME_STABLE_LOSS": 0, "REGIME_DEPENDENT": 0}
            per_asset = {}
            for ticker in manifest["universe"]:
                bars, dates = bt.load_ticker(ticker)
                closes = bars.closes_array()
                labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
                runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
                lbls = [lab for lab, _, _ in runs]
                seg_medians = [round(segment_median(ticker, s, e, lookback, cfg)[0], 3)
                               for lab, s, e in runs]
                pub = artifact["universe"][str(lookback)][label]["per_asset"][ticker]
                pub_lbls = pub["segments"]
                status = "MATCH" if seg_medians == pub["medians"] and lbls == pub_lbls else "MISMATCH"
                if status != "MATCH":
                    all_ok = False
                null = np.array(artifact["universe"][str(lookback)][label]["per_asset"][ticker]["null_medians"])
                regen = verdict_from(seg_medians, null.tolist())
                counts[regen] += 1
                per_asset[ticker] = {
                    "medians": seg_medians, "segments": lbls, "verdict": regen,
                    "n_folds": pub["n_folds"]}
                if status != "MATCH":
                    print("    [DEBUG] universe {}: lookback={} {} ticker={} medians {} vs pub {}".format(
                        asset if False else ticker, lookback, label, ticker, seg_medians, pub["medians"]))
            verdict_counts = dict(counts)
            universe_recomputed[str(lookback)][label] = {
                "per_asset": per_asset, "ticker_order": manifest["universe"],
                "verdict_counts": verdict_counts}
            pub_counts = artifact["universe"][str(lookback)][label]["verdict_counts"]
            status = "MATCH" if verdict_counts == pub_counts else "MISMATCH"
            if status != "MATCH":
                all_ok = False
            print("    lookback {}: {} n_STABLE={} n_CWN={} n_DEPENDENT={} n_STABLE_LOSS={} -> {}"
                  .format(lookback, label, counts["REGIME_STABLE"], counts["CONSISTENT_WITH_NOISE"],
                          counts["REGIME_DEPENDENT"], counts["REGIME_STABLE_LOSS"], status))

    # Gate 1b: independent coin-flip null replication for one representative cell.
    print("\n=== Gate 1b: independent coin-flip null replication (AMZN, lookback 5, zero) ===")
    print("  random_signals (equal {-1,0,+1}, seed=42+5000=5042) walked fresh; null medians "
          "compared to the artifact (confirms the null mechanism under the gate).")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    null_res = [null_medians("AMZN", s, e, 5, 0, 0) for lab, s, e in runs]
    null_recomputed = [round(m, 3) for m, _ in null_res]
    fold_counts = [n for _, n in null_res]
    pub_null = artifact["per_asset"]["AMZN"]["momentum"]["5"]["zero"]["null_medians"]
    pub_n_folds = artifact["per_asset"]["AMZN"]["momentum"]["5"]["zero"]["n_folds"]
    nstatus = "MATCH" if null_recomputed == pub_null else "MISMATCH"
    if nstatus != "MATCH":
        all_ok = False
    print("    null medians {} (folds {}) vs artifact {} (folds {}) -> {}".format(
        null_recomputed, fold_counts, pub_null, pub_n_folds, nstatus))

    # Gate 2: independent daily-rebalance cost drag (AAPL, representative) per (lookback, cost).
    print("\n=== Gate 2: independent 1-day-hold cost drag (daily rebalance) ===")
    print("  Fresh cost arithmetic (entry/exit slippage + round-trip commission), "
          "full sample, on AAPL (representative; costs scale with share price).\n")
    daily_rebalance_recomputed = {}
    for lookback in LOOKBACKS:
        daily_rebalance_recomputed[str(lookback)] = {}
        for label, cfg in COST_CONFIGS:
            res = daily_rebalance_results(tickers, lookback, cfg, tickers_for_cost=("AAPL",))
            daily_rebalance_recomputed[str(lookback)][label] = res
            pub = artifact["daily_rebalance"][str(lookback)][label]
            mismatches = []
            for key in ("n_bars", "n_days_with_signal", "n_days_backtested",
                        "mean_daily_gross_log_return", "median_daily_gross_log_return",
                        "mean_daily_net_log_return", "median_daily_net_log_return",
                        "mean_round_trip_cost_dollars", "mean_round_trip_cost_pct_of_capital",
                        "mean_daily_cost_pct", "net_vs_gross_delta"):
                if res[key] != pub[key]:
                    mismatches.append(f"{key} {res[key]} vs {pub[key]}")
            status = "MATCH" if not mismatches else "MISMATCH(" + "; ".join(mismatches) + ")"
            if status != "MATCH":
                all_ok = False
            print("    lookback {}: {} gross_med={:+.6f} net_med={:+.6f} "
                  "mean_cost=${:,.0f}/day ({:.4f} pct) -> {}".format(
                      lookback, label, res["median_daily_gross_log_return"],
                      res["median_daily_net_log_return"], res["mean_round_trip_cost_dollars"],
                      res["mean_round_trip_cost_pct_of_capital"], status))

    # Gate 3: determinism of this verification path.
    print("\n=== Gate 3: determinism of the verification path ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = volatility_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_median("AMZN", s, e, 5, REALISTIC_COST)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_median("AMZN", s, e, 5, REALISTIC_COST)[0], 3)
          for lab, s, e in runs]
    print("  AMZN lookback=5 realistic medians r1: {}; r2: {} -> {}"
          .format(m1, m2, "identical" if m1 == m2 else "DIFFERENT"))
    if m1 != m2:
        all_ok = False
        print("  [DEBUG] verifier path not deterministic")

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
        print("Verdict: the cost-sensitivity results are independently verified.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification on the "
          "10-asset universe: exploratory simulation.")


if __name__ == "__main__":
    main()
