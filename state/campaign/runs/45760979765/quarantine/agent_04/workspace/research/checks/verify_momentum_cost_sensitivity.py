"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Three verification gates plus a determinism check, all using a fresh code path
(direct ``walk_forward`` + ``noise_benchmark``) that does NOT call
``stress_segments``:

Gate 1  Fresh recomputation of the AMZN/JPM per-segment candidate medians and
        coin-flip null medians for every lookback {3,5,10} and every cost level
        {zero, realistic, conservative}; compared to the check artifact at 6
        decimals. Segment reconstruction, momentum signals and volatility
        blocks are re-implemented here independently.

Gate 2  Cross-consistency: the zero-cost candidate medians for ALL 10 assets x
        {3,5,10} must reproduce the admitted `momentum_lookback_sweep_results.json`
        at 3 decimals (the zero-cost column is the same computation as the
        admitted artifact, so it must match exactly).

Gate 3  Cost-robustness verdict reproducibility: the per-asset conservative
        classification is re-derived from the artifact medians via the
        documented rule, independently of the check's own classification code.

Determinism: AMZN lookback=5 realistic medians recomputed twice are identical.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW, N_BLOCKS, MIN_SEG = 60, 4, 400
LOOKBACKS = (3, 5, 10)
ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_lookback_sweep_results.json"

# Independent copy of the cost model (must match the check's COST_CONFIGS).
COST_CONFIGS = {
    "zero": dict(commission_per_trade=0.0, commission_per_share=0.0,
                 slippage_cents=0.0, slippage_proportional=0.0),
    "realistic": dict(commission_per_trade=2.0, commission_per_share=0.003,
                      slippage_cents=2.0, slippage_proportional=0.0005),
    "conservative": dict(commission_per_trade=4.0, commission_per_share=0.006,
                         slippage_cents=4.0, slippage_proportional=0.0010),
}
COST_ORDER = ("zero", "realistic", "conservative")


def make_cfg(cost_name):
    c = COST_CONFIGS[cost_name]
    return bt.BacktestConfig(
        initial_capital=1e6, warmup_periods=WARM,
        commission_per_trade=c["commission_per_trade"],
        commission_per_share=c["commission_per_share"],
        slippage_cents=c["slippage_cents"],
        slippage_proportional=c["slippage_proportional"],
    )


def vol_blocks(closes, n_blocks, window):
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(float(np.std(np.log(closes[i - window:i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(np.median([v for v in bar_vols[s:e] if v > 0]) if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                runs.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        runs.append((cur, start, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1:i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def seg_candidate_median(seg_bars, seg_signals, cost_name):
    res = bt.walk_forward(seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
                          warmup=WARM, overlap_window=OVERLAP, cfg=make_cfg(cost_name))
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log))


def seg_null_median(seg_bars, lookback, cost_name):
    noise = noise_benchmark(bars=seg_bars, param_grid=[{"lookback": lookback}],
                            train_window=TRAIN, test_window=TEST, warmup=0,
                            overlap_window=0, cfg=make_cfg(cost_name), periods_per_year=252)
    return float(noise.baseline_median_log_return)


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    art = load(ARTIFACT)
    assert art["dataset_id"] == DATASET_ID
    assert art["lookbacks"] == list(LOOKBACKS)
    all_ok = True

    print("=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM; all lookbacks/costs) ===")
    for asset in ["AMZN", "JPM"]:
        bars, _ = bt.load_ticker(asset)
        closes = bars.closes_array()
        runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEG)
        lbls = [lab for lab, _, _ in runs]
        full_sig = momentum_signals(closes, 5)  # signals identical across lookbacks here
        for lb in LOOKBACKS:
            sig = momentum_signals(closes, lb)
            for cost_name in COST_ORDER:
                meds, nulls = [], []
                for lab, s, e in runs:
                    seg_bars = list(bars)[s:e]
                    seg_sig = sig[s:e]
                    meds.append(round(seg_candidate_median(seg_bars, seg_sig, cost_name), 6))
                    nulls.append(round(seg_null_median(seg_bars, lb, cost_name), 6))
                pub = art["per_asset"][asset]["lookback"][str(lb)][cost_name]
                pub_m = [round(v, 6) for v in pub["medians"]]
                pub_n = [round(v, 6) for v in pub["null_medians"]]
                status = ("MATCH" if meds == pub_m and nulls == pub_n
                          and lbls == pub["segments"] else "MISMATCH")
                if status != "MATCH":
                    all_ok = False
                    print("  {} lb={} {}: {} ".format(asset, lb, cost_name, status))
                    print("    recomputed medians {} vs published {}".format(meds, pub_m))
                    print("    recomputed nulls   {} vs published {}".format(nulls, pub_n))
        print("  {}: all lookbacks/costs checked".format(asset))

    print("\n=== Gate 2: zero-cost medians cross-check vs momentum_lookback_sweep_results.json ===")
    sweep = load(SWEEP_ARTIFACT)
    cmismatch = False
    for ticker in sweep["per_asset"]:
        for lb in LOOKBACKS:
            pub = sweep["per_asset"][ticker]["lookback_medians"][str(lb)]
            mine = [round(v, 3) for v in art["per_asset"][ticker]["lookback"][str(lb)]["zero"]["medians"]]
            status = "MATCH" if mine == [round(v, 3) for v in pub] else "MISMATCH"
            if status != "MATCH":
                cmismatch = True
                all_ok = False
                print("  {} lb={}: {} vs {} -> {}".format(ticker, lb, mine, pub, status))
    if not cmismatch:
        print("  all 10 tickers x 3 lookbacks zero-cost medians MATCH the admitted artifact (3dp)")

    print("\n=== Gate 3: conservative cost-robustness verdict reproducibility ===")
    def edge_of(medians, verdict):
        if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
            return "EDGE"
        if verdict == "REGIME_STABLE_LOSS":
            return "LOSS"
        return "NO_EDGE"

    def counts(ticker, cost_name):
        c = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            sr = art["per_asset"][ticker]["lookback"][str(lb)][cost_name]
            c[edge_of(sr["medians"], sr["verdict"])] += 1
        return c

    rmismatch = False
    cons_counts = {"COST_ROBUST": 0, "COST_DEGRADED": 0, "COST_REJECTED": 0}
    for ticker in art["universe"]["ticker_order"]:
        cz = counts(ticker, "zero")
        cc = counts(ticker, "conservative")
        if cz["EDGE"] < 2:
            verdict = "NO_EDGE_ZERO_COST"
        elif cc["EDGE"] >= 2:
            verdict = "COST_ROBUST"
        elif cc["EDGE"] == 1:
            verdict = "COST_DEGRADED"
        else:
            verdict = "COST_REJECTED"
        if verdict in cons_counts:
            cons_counts[verdict] += 1
        pub = art["per_asset"][ticker]["cost_verdict_conservative"]
        status = "MATCH" if verdict == pub else "MISMATCH"
        if status != "MATCH":
            rmismatch = True
            all_ok = False
        print("  {}: zero_edges={} cons_edges={} -> {} vs published {} -> {}".format(
            ticker, cz["EDGE"], cc["EDGE"], verdict, pub, status))
    derived_class = ("COST_ROBUST" if cons_counts["COST_ROBUST"] >= 4
                     else ("COST_DEGRADED" if cons_counts["COST_ROBUST"] >= 1 else "COST_REJECTED"))
    pub_class = art["universe"]["classification_conservative"]
    print("  universe conservative class: derived {} vs published {} -> {}".format(
        derived_class, pub_class, "MATCH" if derived_class == pub_class else "MISMATCH"))
    if derived_class != pub_class:
        all_ok = False

    print("\nDeterminism: AMZN lookback=5 realistic medians recomputed twice.")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    runs = segments_from_labels(vol_blocks(closes, N_BLOCKS, WINDOW), MIN_SEG)
    sig = momentum_signals(closes, 5)
    m1 = [round(seg_candidate_median(list(bars)[s:e], sig[s:e], "realistic"), 6)
          for lab, s, e in runs]
    m2 = [round(seg_candidate_median(list(bars)[s:e], sig[s:e], "realistic"), 6)
          for lab, s, e in runs]
    det = m1 == m2
    print("  r1: {}; r2: {} -> {}".format(m1, m2, "identical" if det else "DIFFERENT"))
    if not det:
        all_ok = False

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact (candidate + null, "
              "all lookbacks/costs on AMZN/JPM), the zero-cost column cross-checks against "
              "momentum_lookback_sweep_results.json, and the conservative cost-robustness "
              "verdicts reproduce.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the independent "
              "path. Investigate before recording results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production execution.")


if __name__ == "__main__":
    main()