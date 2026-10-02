"""Minimal deterministic backtest + synthetic-data toolkit.

Research / simulation only. No live trading.
Stdlib + numpy only.

Structure:
- data.py           : synthetic daily bar generator (tooling validation only)
- engine.py          : event-driven backtest with walk-forward IS/OOS support
- metrics.py         : deterministic performance/risk metrics
- leakage.py         : look-ahead and equity-fill integrity checks
- perturbation.py    : parameter-sensitivity / robustness testing
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
    "LeakySignalError",
    "check_signal_integrity",
    "check_equity_matches_fills",
    "ParameterSet",
    "SweepResult",
    "SweepSummary",
    "parameter_sweep",
    "parameter_grid_around",
    "random_signals",
    "noise_benchmark",
    "sweep_summary",
]
