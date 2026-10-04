"""Minimal deterministic backtest + synthetic-data toolkit.

Research / simulation only. No live trading.
Stdlib + numpy only.

Structure:
- data.py           : synthetic daily bar generator (tooling validation only)
- engine.py          : event-driven backtest with walk-forward IS/OOS support
- metrics.py         : deterministic performance/risk metrics
- leakage.py         : look-ahead and equity-fill integrity checks
- perturbation.py    : parameter-sensitivity / robustness testing
- universe.py        : asset-universe robustness / concentration testing
"""
from __future__ import annotations

__version__ = "0.1.0"

from .data import BarSequence, Regime, generate_bars
from .engine import (
    Backtest,
    BacktestConfig,
    Bar,
    Fill,
    FillResult,
    FoldResult,
    MarginCall,
    Order,
    Signal,
    WalkForwardResult,
    run_bars,
    walk_forward,
)
from .leakage import LeakySignalError, check_equity_matches_fills, check_signal_integrity
from .metrics import Metrics, compute_metrics, TradeSummary
from .perturbation import (
    ParameterSet,
    SweepResult,
    SweepSummary,
    noise_benchmark,
    parameter_grid_around,
    parameter_sweep,
    random_signals,
    sweep_summary,
)
from .regime_stability import (
    EdgeFree,
    EdgePresent,
    OUTLIER_TOL,
    RegimeScenario,
    RegimeScenarioResult,
    RegimeStressResult,
    active_when_anything,
    active_when_turbulent,
    canonical_regime_scenarios,
    conditional_signal,
    deterministic_edge_signal,
    generate_under,
    regime_stress,
    run_scenario,
    stress_regime_scenarios,
    direction_signal,
    always_long_signal,
    stress_segments,
    segment_fn_from_labels,
    volatility_segments,
    ma_crossover_signals,
    volatility_blocks,
    RegimeUniverseSummary,
    stress_segments_across_tickers,
)
from .universe import (
    ASSET_VERDICT_CONCENTRATED,
    ASSET_VERDICT_CONSISTENT,
    ASSET_VERDICT_NO_EDGE,
    AssetSweepResult,
    AssetSweepSummary,
    sweep_across_assets,
    asset_sweep_summary,
    uniform_regime_assets,
    flat_regime_assets,
)
from .real_data import load_ticker, load_universe

__all__ = [
    "__version__",
    "generate_bars",
    "Regime",
    "BarSequence",
    "Backtest",
    "BacktestConfig",
    "Bar",
    "Signal",
    "Order",
    "Fill",
    "FillResult",
    "run_bars",
    "walk_forward",
    "FoldResult",
    "WalkForwardResult",
    "Metrics",
    "compute_metrics",
    "TradeSummary",
    "MarginCall",
    "LeakySignalError",
    "check_signal_integrity",
    "check_equity_matches_fills",
    "ParameterSet",
    "SweepResult",
    "SweepSummary",
    "OUTLIER_TOL",
    "EdgeFree",
    "EdgePresent",
    "ParameterSet",
    "SweepResult",
    "SweepSummary",
    "parameter_sweep",
    "parameter_grid_around",
    "random_signals",
    "noise_benchmark",
    "sweep_summary",
    "ASSET_VERDICT_CONCENTRATED",
    "ASSET_VERDICT_CONSISTENT",
    "ASSET_VERDICT_NO_EDGE",
    "AssetSweepResult",
    "AssetSweepSummary",
    "sweep_across_assets",
    "asset_sweep_summary",
    "uniform_regime_assets",
    "flat_regime_assets",
    "stress_segments",
    "segment_fn_from_labels",
    "volatility_segments",
    "ma_crossover_signals",
    "volatility_blocks",
    "RegimeUniverseSummary",
    "stress_segments_across_tickers",
    "load_ticker",
    "load_universe",
]
