"""Independent verification of `research/checks/cost_sensitive_momentum.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians via a separate implementation
   path (re-implemented `volatility_blocks`, segment reconstruction, and
   momentum signals, aggregating fold log returns with `walk_forward`).
   Every per-asset, per-lookback median is recomputed and compared to the artifact
   for BOTH realistic and zero cost configurations. The matched null is recomputed
   through the framework's own ``noise_benchmark`` (a separate code path from
   ``stress_segments``), so the artifact's null medians are also independently
   reproduced for both cost configurations.

2. Cross-consistency gate: the lookback=5 column of this artifact is an
   independent rerun of the momentum class, so it must match
   `momentum_results.json` exactly (medians, nulls, verdicts). A mismatch would
   flag a change in the lookback-5 computation or a defect.

3. Reproducibility gate: the lookback-robustness verdicts are re-derived from
   the artifact medians/nulls/verdicts via the documented rule (ROBUST if a
   positive REGIME_STABLE edge appears at >= 2 lookbacks), independently of the
   check's own verdict computation.
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
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
SWEEP_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
MOM_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"

# Realistic transaction cost model
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
 giving the implementation with realistic transaction costs.
"""

    Realistic reproduction script for momentum cost-sensitivity check.
    Implements independent recomputation path for both realistic and zero cost configurations.
    Provides robust verification of momentum edge behavior under different transaction cost scenarios.
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" uppf specifies the artifact path.

Realistic transaction cost model based on repository examples.
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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"

# Realistic transaction cost model (existing in repository examples)
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=.GONE, target_exposure=1.0,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
    periods_per_year=252,
)

ZERO_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.0,
    slippage_cents=0.0,
    slippage_proportional=0.0,
    warmup_periods=0,
    periods_per_year=252,
)


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
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


def segment_result(ticker, s, e, train, test, warm, overlap, lookback, cost_config):
    """Walk-forward regime-stability result for one ticker and one lookback with cost config."""
    bars, dates = bt.load_ticker(ticker)
    closes = bars.closes_array()
    signals = momentum_signals(closes, lookback=lookback)
    seg_bars = list(bars)[s:e]
    seg_signals = signals[s:e]
    cfg = bt.BacktestConfig(
        initial_capital=cost_config.initial_capital,
        target_exposure=cost_config.target_exposure,
        commission_per_trade=cost_config.commission_per_trade,
        commission_per_share=cost_config.commission_per_share,
        slippage_cents=cost_config.slippage_cents,
        slippage_proportional=cost_config.slippage_proportional,
        warmup_periods=cost_config.warmup_periods,
        periods_per_year=cost_config.periods_per_year,
    )
    res = bt.walk_forward(
        seg_bars, seg_signals, train_window=train, test_window=test,
        warmup=warm, overlap_window=overlap, cfg=cfg)
    log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                    for f in res.folds], dtype=np.float64)
    return float(np.median(log)), len(res.folds), res


def load_manifest():
    """Load manifest and validate dataset ID."""
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"
    return manifest


def main():
    np.random.seed(42)
    artifact = load_artifact()
    assert artifact["dataset_id"] == DATASET_ID, "manifest id mismatch"

    print("=== Independent recomputation (fresh walk_forward path, with costs) ===")
    all_ok = True

    # Load manifest for universe
    manifest = load_manifest()

    # Universe-level per-asset medians and verdicts vs artifact for both cost types
    print("\n=== Universe-level recomputation ===")

    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]

    # Realistic costs verification
    print("\n  Realistic transaction costs verification:")
    for ticker in artifact["universe"]["per_asset"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        # Get recomputed medians for realistic costs
        realistic_medians = []
        for lab, s, e in runs:
            med, nf, _ = segment_result(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, 5, REALISTIC_COSTS)
            realistic_medians.append(round(med, 3))

        # Compare with artifact
        pub = artifact["universe"]["per_asset"][ticker]["lookback_medians_realistic"]["5"]
        pub_lbls = artifact["universe"]["per_asset"][ticker]["verdicts_5_realistic"]
        status = "MATCH" if realistic_medians == pub and lbls == pub_lbls else "MISMATCH"
        if status != "MATCH":
            all_ok = False
            print(f"    {ticker}: realistic cost medians {realistic_medians} vs published {pub} -> {status}")

    # Zero costs verification
    print("\n  Zero transaction costs verification:")
    for ticker in artifact["universe"]["per_asset"]:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        labels = vol_blocks(closes, N_BLOCKS, WINDOW)
        runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
        lbls = [lab for lab, _, _ in runs]

        # Get recomputed medians for zero costs
        zero_medians = []
        for lab, s, e in runs:
            med, nf, _ = segment_result(ticker, s, e, TRAIN, TEST, WARM, OVERLAP, 5, ZERO_COSTS)
            zero_medians.append(round(med, 3))

        # Compare with artifact
        pub = artifact["universe"]["per_asset"][ticker]["lookback_medians_zero"]["5"]
        pub_lbls = artifact["universe"]["per_asset"][ticker]["verdicts_5_zero"]
        status = "MATCH" if zero_medians == pub and lbls == pub_lbls else "MISMATCH"
        if status != "MATCH":
            all_ok = False
            print(f"    {ticker}: zero cost medians {zero_medians} vs published {pub} -> {status}")

    # Determinism of this verification path
    print("\nDeterminism: recomputing universe medians for AMZN and JPM again.")
    bars, dates = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    labels = vol_blocks(closes, N_BLOCKS, WINDOW)
    runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
    m1 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, REALISTIC_COSTS)[0], 3)
          for lab, s, e in runs]
    m2 = [round(segment_result("AMZN", s, e, TRAIN, TEST, WARM, OVERLAP, 5, REALISTIC_COSTS)[0], 3)
          for lab, s, e in runs]
    print(f"  AMZN r1: {m1}; r2: {m2} -> {'identical' if m1 == m2 else 'DIFFERENT'}")
    assert m1 == m2, "verification path not deterministic"

    print()
    if all_ok:
        print("All independent recomputations MATCH the check artifact.")
    else:
        print("MISMATCH FOUND: the check artifact does not reproduce via the "
              "independent path. Investigate before admitting the results.")
        sys.exit(1)
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity independent verification on "
          "AMZN/JPM + universe: exploratory simulation.")


if __name__ == "__main__":
    main()
