"""Debug: why does the verifier's coin-flip null differ from the framework null?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
N_BLOCKS = 4
WINDOW = 60
MIN_SEGMENT_BARS = 400

ZERO = bt.BacktestConfig()


def vol_blocks(closes, n_blocks, window):
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


def coin_flip_signals(n, seed):
    rng = np.random.default_rng(seed)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]


def param_seed_for_lookback(lookback, base_seed=42):
    return base_seed + int(round(float(lookback) * 1000))


def load_artifact():
    with open(Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json") as f:
        return json.load(f)


def main():
    m = load_manifest()
    artifact = load_artifact()

    print("=== AMZN lookback=3 zero (3 segments) ===")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    print("segments:", runs)

    for seg_name, s, e in runs:
        seg_bars = list(bars)[s:e]
        n = len(seg_bars)
        pseed = param_seed_for_lookback(3)

        # 1) Framework path: bt.noise_benchmark (same code path that built the artifact)
        fw_null = bt.noise_benchmark(
            seg_bars, param_grid=[{"lookback": 3.0}], train_window=TRAIN, test_window=TEST,
            warmup=0, overlap_window=0, cfg=ZERO, periods_per_year=252)
        fw_med = round(fw_null.baseline_median_log_return, 3)
        fw_folds = [round(float(x), 3) for x in fw_null.noise_fold_median_log_returns]
        fw_seed = param_seed_for_lookback(3)

        # 2) Verifier path: fresh coin-flip walk_forward loop
        sigs = coin_flip_signals(n, pseed)
        print("coin-flip weights (first 12):", [round(w.weight, 2) for w in sigs[:12]])
        res = bt.walk_forward(
            seg_bars, sigs, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=ZERO)
        ver_med = round(float(np.median([
            np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in res.folds
        ])), 3)
        ver_folds = [round(float(np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))), 3)
                     for f in res.folds]

        pub = artifact["per_asset"]["AMZN"]["momentum"]["3"]["zero"]
        pub_null = pub["null_medians"]
        print(f"  seg={seg_name} n_bars={n} pseed={pseed}")
        print(f"  framework null median : {fw_med} folds={fw_folds}")
        print(f"  verifier null median  : {ver_med} folds={ver_folds}")
        print(f"  artifact null median  : {pub_null}")
        print(f"  framework==artifact: {fw_med == pub_null[runs.index((seg_name, s, e))]}, "
              f"verifier==artifact: {ver_med == pub_null[runs.index((seg_name, s, e))]}")

    print("\n=== NVDA lookback=5 realistic (verifier-reported oddity) ===")
    bars, dates = bt.load_ticker("NVDA")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    print("segments:", runs)
    for seg_name, s, e in runs:
        seg_bars = list(bars)[s:e]
        n = len(seg_bars)
        pseed = param_seed_for_lookback(5)
        sigs = coin_flip_signals(n, pseed)
        res = bt.walk_forward(
            seg_bars, sigs, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=REALISTIC if (REALISTIC := bt.BacktestConfig(
                commission_per_trade=2.0, commission_per_share=0.003,
                slippage_cents=2.0, slippage_proportional=0.0005)) else None)
        log = [np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None)) for f in res.folds]
        print(f"  seg={seg_name} n_bars={n} n_folds={len(res.folds)} "
              f"median={round(float(np.median(log)), 3)} "
              f"first3={log[:3] if len(log) else []}")


if __name__ == "__main__":
    sys.exit(main())
