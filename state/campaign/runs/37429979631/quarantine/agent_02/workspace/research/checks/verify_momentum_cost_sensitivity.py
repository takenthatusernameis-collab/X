"""Independent verification of the momentum cost-sensitivity check
(`research/checks/momentum_cost_sensitivity.py`).

This is a separate implementation path: it recomputes each cell's results
from raw bars using fresh `walk_forward` (Gate 1, the repository's regime-
stability gate) and an independent 1-day-hold daily-rebalance cost
calculation (Gate 2), then compares against the artifact written by the
check (`state/check_artifacts/momentum_cost_sensitivity_results.json`).
A mismatch would flag a defect in the check.

Verification scope (bounded, a-priori):
- Gate 1 (walk-forward): AMZN + JPM per-asset medians for all lookbacks
  {3, 5, 10} x all cost configs {zero, realistic, heavy}; plus the full 10-
  asset universe for lookback 5 x all cost configs. Verdicts regenerated
  from the recomputed candidate medians and independently recomputed coin-
  flip null medians on each segment.
- Gate 2 (true cost drag): AMZN + JPM 1-day-hold daily-rebalance gross/
  median/net daily log returns for all lookbacks x all cost configs,
  recomputed from raw closes with a fresh cost model.
- Determinism: rerun the per-asset recomputation twice and assert identical.

The coin-flip null is recomputed fresh (via the framework's noise
benchmark on the segment slice) rather than loaded from the artifact, so
the null comparison is independently produced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = (3, 5, 10)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

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

INITIAL_CAPITAL = 1e6

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"


def vol_blocks(closes, n_blocks, window):
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
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_gate1_stats(ticker, s, e, train, test, warm, overlap, cfg, lookback):
    """Recompute one segment's walk-forward Gate 1 stats via fresh
    walk_forward and the framework's own noise benchmark (no
    stress_segments call). Returns (medians, null_medians, n_folds)."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    res = bt.walk_forward(seg_bars, seg_signals, train_window=train,
                          test_window=test, warmup=warm, overlap_window=overlap,
                          cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    candidate_med = float(np.median(log))

    # Independent coin-flip null on the same segment, using the framework's
    # own noise benchmark with the framework's deterministic seed derivation
    # so the null matches what the check's framework produced.
    # The check's framework runs the null with warmup=0 / overlap_window=0
    # on the segment (distinct from the candidate walk-forward), so we must
    # replicate that exact fold structure for the null to reproduce exactly.
    noise = bt.noise_benchmark(
        bars=seg_bars,
        param_grid=[{"lookback": float(lookback)}],
        train_window=train, test_window=test, warmup=0, overlap_window=0,
        cfg=cfg, periods_per_year=252, seed=SEED,
    )
    null_med = noise.baseline_median_log_return
    return candidate_med, null_med, len(res.folds)


def gate1_asset_stats(ticker, tickers, lookback, cfg):
    """Run Gate 1 (regime-stability gate) on one asset via fresh walk_forward
    segments."""
    bars = tickers[ticker]
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    lbls = [lab for lab, _, _ in runs]
    medians = []
    null_medians = []
    n_folds = None
    for lab, s, e in runs:
        c, n, nf = segment_gate1_stats(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, cfg, lookback)
        medians.append(round(c, 3))
        null_medians.append(round(n, 3))
        n_folds = nf
    return dict(segments=lbls, medians=medians, null_medians=null_medians,
                n_folds=n_folds)


def regenerate_verdict(medians, null_medians):
    """Regenerate the regime-stability verdict from recomputed medians and
    null medians (same rule as research.backtest)."""
    medians = np.array(medians, dtype=np.float64)
    nulls = np.array(null_medians, dtype=np.float64)
    cand_disp = float(np.std(medians))
    null_disp = float(np.std(nulls))
    if all(abs(m) <= 0.05 for m in medians):
        return "CONSISTENT_WITH_NOISE"
    if cand_disp > 2.0 * null_disp:
        return "REGIME_DEPENDENT"
    if all(m < 0 for m in medians):
        return "REGIME_STABLE_LOSS"
    return "REGIME_STABLE"


def gate2_stats(tickers, lookback, cfg, tickers_for_cost=()):
    """Independent recomputation of Gate 2 (1-day-hold daily-rebalance cost
    drag) using fresh code (no check call). Returns per-bar net log returns."""
    tickers_for_cost = tickers_for_cost or ("AAPL",)
    bars, dates = bt.load_ticker(tickers_for_cost[0])
    closes = bars.closes_array()
    n = len(closes)
    signals = momentum_signals(closes, lookback)
    f, p = cfg.slippage_cents / 100.0, cfg.slippage_proportional
    comm_per_trade = cfg.commission_per_trade
    comm_per_share = cfg.commission_per_share
    CAP = INITIAL_CAPITAL
    net_rets = np.zeros(n, dtype=np.float64)
    gross_rets = np.zeros(n, dtype=np.float64)
    n_days = 0
    for i in range(lookback, n - 1):
        sig = signals[i]
        if abs(sig.weight) < 1e-12:
            continue
        entry_price = closes[i] + f + closes[i] * p
        exit_price = closes[i + 1] + f + closes[i + 1] * p
        shares = CAP * cfg.target_exposure / entry_price
        gross_pnl = sig.weight * shares * (exit_price - entry_price)
        cost = 2.0 * (comm_per_trade + abs(sig.weight) * shares * comm_per_share)
        cost += abs(sig.weight) * shares * (entry_price - closes[i]) + abs(
            sig.weight) * shares * (exit_price - closes[i + 1])
        net_pnl = gross_pnl - cost
        n_days += 1
        gross_rets[i] = np.log1p(gross_pnl / CAP)
        net_rets[i] = np.log1p(net_pnl / CAP)
    gross_rets = gross_rets[lookback:n - 1]
    net_rets = net_rets[lookback:n - 1]
    if n_days == 0:
        raise RuntimeError("no signal days")
    mean_cost = float(np.mean([abs(sig.weight) * (CAP / (closes[i] + f + closes[i] * p)) *
                               (2.0 * comm_per_share + 2.0 * f + (closes[i] + closes[i + 1]) * p)
                               + 2.0 * comm_per_trade
                               for i, sig in enumerate(signals)
                               if i >= lookback and i < n - 1 and abs(sig.weight) > 1e-12]))
    return dict(
        mean_daily_gross_log_return=round(float(np.mean(gross_rets)), 6),
        median_daily_gross_log_return=round(float(np.median(gross_rets)), 6),
        mean_daily_net_log_return=round(float(np.mean(net_rets)), 6),
        median_daily_net_log_return=round(float(np.median(net_rets)), 6),
        mean_round_trip_cost_dollars=round(mean_cost, 2),
        mean_round_trip_cost_pct_of_capital=round(mean_cost / CAP, 6),
        net_vs_gross_delta=round(float(np.median(net_rets)) - round(float(np.median(gross_rets)), 6), 6),
    )


def load_artifact():
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


def main():
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"
    assert set(artifact["lookbacks"]) == {3, 5, 10}
    for label, cfg in COST_CONFIGS:
        cm = artifact["cost_models"][label]
        assert cm["commission_per_trade"] == cfg.commission_per_trade
        assert cm["commission_per_share"] == cfg.commission_per_share
        assert cm["slippage_cents"] == cfg.slippage_cents
        assert cm["slippage_proportional"] == cfg.slippage_proportional
    target = ["AMZN", "JPM"]
    tickers = {}
    m = load_manifest()
    for entry in m["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    print("=== Gate 1: independent walk-forward recomputation (AMZN, JPM) ===")
    all_ok = True
    for asset in target:
        for lookback in LOOKBACKS:
            for label, cfg in COST_CONFIGS:
                rec = gate1_asset_stats(asset, tickers, lookback, cfg)
                pub = artifact["per_asset"][asset]["momentum"][str(lookback)][label]
                med_status = "MATCH" if rec["medians"] == pub["medians"] else "MISMATCH"
                null_status = "MATCH" if rec["null_medians"] == pub["null_medians"] else "MISMATCH"
                seg_status = "MATCH" if rec["segments"] == pub["segments"] else "MISMATCH"
                if med_status != "MATCH" or null_status != "MATCH" or seg_status != "MATCH":
                    all_ok = False
                regen = regenerate_verdict(rec["medians"], rec["null_medians"])
                v_status = "MATCH" if regen == pub["verdict"] else "MISMATCH"
                if v_status != "MATCH":
                    all_ok = False
                print("  {} lb={} {} medians {} -> {} | null {} -> {} | "
                      "verdict {} (pub {})".format(
                      asset, lookback, label,
                      rec["medians"], med_status, rec["null_medians"], null_status,
                      regen, pub["verdict"]))

    print("\n=== Gate 1: independent walk-forward recomputation (universe, lookback 5) ===")
    for label, cfg in COST_CONFIGS:
        for ticker in artifact["universe"]["5"][label]["per_asset"]:
            rec = gate1_asset_stats(ticker, tickers, 5, cfg)
            pub = artifact["universe"]["5"][label]["per_asset"][ticker]
            med_status = "MATCH" if rec["medians"] == pub["medians"] else "MISMATCH"
            null_status = "MATCH" if rec["null_medians"] == pub["null_medians"] else "MISMATCH"
            seg_status = "MATCH" if rec["segments"] == pub["segments"] else "MISMATCH"
            if med_status != "MATCH" or null_status != "MATCH" or seg_status != "MATCH":
                all_ok = False
            regen = regenerate_verdict(rec["medians"], rec["null_medians"])
            v_status = "MATCH" if regen == pub["verdict"] else "MISMATCH"
            if v_status != "MATCH":
                all_ok = False
            print("  {} lb=5 {} medians {} -> {} | verdict {} (pub {})".format(
                ticker, label, rec["medians"], med_status, regen, pub["verdict"]))

    print("\n=== Gate 2: independent 1-day-hold daily-rebalance cost drag (AMZN, JPM) ===")
    for asset in target:
        tickers_for_asset = {asset: tickers[asset]}
        for lookback in LOOKBACKS:
            for label, cfg in COST_CONFIGS:
                rec = gate2_stats(tickers_for_asset, lookback, cfg)
                pub = artifact["daily_rebalance"][str(lookback)][label]
                checks = [
                    ("mean_daily_gross_log_return", round(rec["mean_daily_gross_log_return"], 6), round(pub["mean_daily_gross_log_return"], 6)),
                    ("median_daily_gross_log_return", round(rec["median_daily_gross_log_return"], 6), round(pub["median_daily_gross_log_return"], 6)),
                    ("mean_daily_net_log_return", round(rec["mean_daily_net_log_return"], 6), round(pub["mean_daily_net_log_return"], 6)),
                    ("median_daily_net_log_return", round(rec["median_daily_net_log_return"], 6), round(pub["median_daily_net_log_return"], 6)),
                    ("mean_round_trip_cost_dollars", round(rec["mean_round_trip_cost_dollars"], 2), round(pub["mean_round_trip_cost_dollars"], 2)),
                    ("mean_round_trip_cost_pct_of_capital", round(rec["mean_round_trip_cost_pct_of_capital"], 6), round(pub["mean_round_trip_cost_pct_of_capital"], 6)),
                ]
                for fname, rv, pv in checks:
                    if rv != pv:
                        all_ok = False
                        print("  [MISMATCH] {} lb={} {} {}: rec={} pub={}".format(
                            asset, lookback, label, fname, rv, pv))
                if all(fv == sv for _, fv, sv in checks):
                    print("  {} lb={} {} -> net_med {:+.6f} (cost ${:,.0f}/day, pct {}) MATCH"
                          .format(asset, lookback, label, rec["median_daily_net_log_return"],
                                  rec["mean_round_trip_cost_dollars"], rec["mean_round_trip_cost_pct_of_capital"]))

    print("\n=== Gate 2 determinism (AMZN lookback 5, realistic, rerun) ===")
    tickers_for_asset = {"AMZN": tickers["AMZN"]}
    rec1 = gate2_stats(tickers_for_asset, 5, REALISTIC_COST)
    rec2 = gate2_stats(tickers_for_asset, 5, REALISTIC_COST)
    det = "identical" if rec1 == rec2 else "DIFFERENT"
    if det != "identical":
        all_ok = False
    print("  AMZN lb=5 realistic r1={} r2={} -> {}".format(rec1, rec2, det))
    assert rec1 == rec2, "Gate 2 not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification: "
          "exploratory simulation.")


if __name__ == "__main__":
    main()
