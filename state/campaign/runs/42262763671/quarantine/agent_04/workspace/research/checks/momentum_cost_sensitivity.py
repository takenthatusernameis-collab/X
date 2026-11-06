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

# Realistic cost model matching the repository's default BacktestConfig
cost_config = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=0.0,
    commission_per_share=0.003,
    slippage_cents=5.0,
    slippage_proportional=0.001,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def segment_result(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback\n    with realistic transaction costs applied.\n\n    Uses the framework's walk_forward for the candidate, and the same segments\n    but with a matched coin-flip null benchmark to compute the null median.\n    """
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)

    # Create segments using the same volatility blocks as the existing framework\n    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)\n    seg_fn = bt.segment_fn_from_labels(labels)\n
    # Generate momentum signals for this lookback\n    sig = bt.momentum_signals(closes, lookback=lookback)\n    if len(sig) != len(closes):\n        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")\n\n    # Run walk-forward with realistic costs\n    res = bt.stress_segments(\n        signals_fn=bt.momentum_signals,\n        bars=bars,\n        signals=sig,\n        param_grid=[{"lookback": lookback}],\n        baseline=(("lookback", lookback),),\n        segment_fn=seg_fn,\n        train_window=TRAIN,\n        test_window=TEST,\n        warmup=WARM,\n        overlap_window=OVERLAP,\n        cfg=cost_config,  # Apply realistic costs here via BacktestConfig\n        periods_per_year=252,\n        min_segment_bars=MIN_SEGMENT_BARS,\n    )\n\n    scen = res.scenarios[0]\n    return dict(\n        segments=[s.name for s in res.scenarios],\n        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],\n        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],\n        candidate_dispersion=round(res.candidate_dispersion, 3),\n        null_dispersion=round(res.null_dispersion, 3),\n        n_folds=res.n_folds,\n        verdict=res.overall_verdict,\n    )