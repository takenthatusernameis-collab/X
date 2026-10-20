"""Parameter-sensitivity / robustness testing for walk-forward backtests.

Research/simulation only. Not live trading.

A robust strategy should not depend on one exact parameter value. This
module runs the same walk-forward validation across a grid of parameters
and summarizes how out-of-sample performance behaves as parameters move
away from a baseline. The output is a falsification-oriented report: on
pure noise the results are flat and center on zero; a spurious edge
collapses once the parameter changes.

Usage pattern::

    bars = generate_bars(2500, seed=1)

    def signals(closes, fast, slow):
        # past-only MA signals ...
        ...

    grid = parameter_grid_around(
        (("fast", 20), ("slow", 60)),
        multipliers=(0.5, 1.0, 2.0),
    )
    result = parameter_sweep(
        signals_fn=signals,
        bars=list(bars),
        param_grid=grid,
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
    )
    summary = sweep_summary(result, baseline=(("fast", 20), ("slow", 60)))
    summary.inspect()
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

import numpy as np

from .data import BarSequence
from .engine import BacktestConfig, Signal, walk_forward


ParameterSet = Tuple[Tuple[str, Any], ...]
"""Hashable mapping of parameter name to value, in a fixed order."""


@dataclass(frozen=True)
class SweepResult:
    """Results of one parameter sweep."""

    param_sets: List[ParameterSet]
    fold_total_returns: List[List[float]]  # per param set, then per fold
    train_window_bars: int
    test_window_bars: int
    n_folds: int
    n_periods: int


@dataclass(frozen=True)
class SweepSummary:
    """Summarized robustness output for one sweep."""

    param_sets: List[ParameterSet]
    baseline_param_set: ParameterSet
    median_log_returns: List[float]  # per param set, same order as param sets
    deviations: List[float]          # ascending, max relative deviation from baseline
    medians_at_deviation: Dict[float, float]
    n_param_sets: int
    n_positive: int
    n_zero: int
    n_negative: int
    fraction_positive: float
    best_param_set: ParameterSet
    best_median_log_return: float
    worst_param_set: ParameterSet
    worst_median_log_return: float
    noise_median_log_return: Optional[float]
    n_folds: int
    n_periods: int
    train_window_bars: int
    test_window_bars: int
    noise_fold_median_log_returns: Optional[Sequence[float]] = None
    """Fold-level log returns of the coin-flip null's baseline parameter set
    (same ``n_folds``), when the summary was produced from a noise benchmark.
    Used by ``compare_noise`` to compute a sample-calibrated tolerance (2x the
    null's fold dispersion divided by sqrt(n_folds)) instead of the fixed 0.05
    default, and by ``effective_tolerance`` to expose that tolerance."""

    @property
    def effective_tolerance(self) -> float:
        """The tolerance ``compare_noise`` uses when called with no explicit ``tol``."""
        fold_returns = list(self.noise_fold_median_log_returns or [])
        if len(fold_returns) >= 2:
            return 2 * float(np.std(fold_returns, ddof=1)) / math.sqrt(len(fold_returns))
        return 0.05

    @property
    def baseline_median_log_return(self) -> float:
        idx = next(i for i, ps in enumerate(self.param_sets) if ps == self.baseline_param_set)
        return self.median_log_returns[idx]

    def param_set_at(self, deviation: float) -> Optional[float]:
        """Median log return for the exact deviation level, if any."""
        return self.medians_at_deviation.get(deviation)

    def compare_noise(
        self,
        noise_median: float,
        noise_fold_median_log_returns: Optional[Sequence[float]] = None,
        tol: float = 0.05,
    ) -> bool:
        """Return True if the baseline is within `tol` of the noise median.

        When no explicit `tol` is passed and the null's fold-level log
        returns are available (from a noise benchmark), the tolerance is
        computed as 2x the coin-flip null's fold dispersion at the same
        sample size, i.e. 2 * null_std / sqrt(n_folds), which is twice the
        standard error of the null's median. This makes the pass/fail
        decision depend on the framework's own noise at the observed sample
        size instead of a fixed 0.05 band. Pass `tol=` explicitly to force a
        fixed tolerance.
        """
        if tol is None:
            fold_returns = (
                noise_fold_median_log_returns or self.noise_fold_median_log_returns
            )
            if fold_returns is not None and len(fold_returns) >= 2:
                tol = 2 * float(np.std(fold_returns, ddof=1)) / math.sqrt(
                    len(fold_returns)
                )
        return abs(self.baseline_median_log_return - noise_median) <= tol

    def inspect(self) -> str:
        lines = [
            "=== Parameter-sensitivity / robustness summary ===",
            "",
            f"Baseline: {self.baseline_param_set}",
            f"Folds: {self.n_folds}  Periods: {self.n_periods}",
            f"Train window: {self.train_window_bars}d  Test window: {self.test_window_bars}d",
            "",
            "Median log return per parameter set:",
        ]
        for ps, med in zip(self.param_sets, self.median_log_returns):
            lines.append(f"  {ps}: {med:+.3f}")
        lines.append("")
        lines.append("Deviation scan (median log return across parameter sets at each")
        lines.append("max relative deviation from the baseline):")
        for dev, med in self.medians_at_deviation.items():
            marker = "  <-- baseline" if abs(dev) < 1e-12 else ""
            lines.append(f"  {dev:4.2f}x:  {med:+.3f}{marker}")
        lines.append("")
        lines.append(
            f"Median log return positive / zero / negative across parameter sets: "
            f"{self.n_positive} / {self.n_zero} / {self.n_negative} "
            f"({self.fraction_positive:+.1%} positive)"
        )
        lines.append("")
        lines.append(f"Best parameter set: {self.best_param_set}  median log return {self.best_median_log_return:+.3f}")
        lines.append(f"Worst parameter set: {self.worst_param_set}  median log return {self.worst_median_log_return:+.3f}")
        if self.noise_median_log_return is not None:
            near = "yes" if self.compare_noise(self.noise_median_log_return) else "no"
            lines.append("")
            lines.append(f"Null (coin-flip) benchmark: median log return {self.noise_median_log_return:+.3f}")
            lines.append(
                f"Baseline within {self.effective_tolerance:+.3f} of null "
                f"(sample-calibrated 2x null std / sqrt(n_folds)): {near}"
            )
        lines.append("")
        lines.append("Interpretation: a robust strategy shows little or no systematic")
        lines.append("decline in median log return as parameters deviate from the")
        lines.append("baseline and a stable sign of median performance across the grid.")
        return "\n".join(lines)


def parameter_sweep(
    signals_fn,
    bars,
    param_grid: List[Dict[str, Any]],
    train_window: int,
    test_window: int,
    warmup: int = 0,
    overlap_window: int = 0,
    cfg: BacktestConfig = BacktestConfig(),
    periods_per_year: int = 252,
) -> SweepResult:
    """Run walk-forward backtests for every parameter set in the grid.

    Args:
        signals_fn: callable(closes: np.ndarray, **params) returning a list
            of Signal with the same length as `bars`. Must be past-only: it
            may use only closes up to and including each signal bar's date
            and must not touch open/high/low.
        bars: full bar series (BarSequence or iterable of Bar).
        param_grid: list of parameter sets; each is a dict of name -> value.
        train_window, test_window, warmup, overlap_window, cfg, periods_per_year:
            forwarded to walk_forward.

    Returns:
        SweepResult with per-fold total returns for every parameter set.
    """
    if not param_grid:
        raise ValueError("param_grid must contain at least one parameter set")

    if isinstance(bars, BarSequence):
        closes = bars.closes_array()
        bars_list = list(bars)
    else:
        closes = np.array([float(b.close) for b in bars], dtype=np.float64)
        bars_list = list(bars)
    n = len(closes)

    fold_returns: List[List[float]] = []
    for params in param_grid:
        signals = signals_fn(closes, **params)
        if len(signals) != n:
            raise ValueError(
                f"signals_fn returned {len(signals)} signals for {n} bars"
            )
        result = walk_forward(
            bars=bars_list,
            signals=signals,
            train_window=train_window,
            test_window=test_window,
            warmup=warmup,
            overlap_window=overlap_window,
            cfg=cfg,
        )
        fold_returns.append([float(f.metrics["total_return"]) for f in result.folds])

    param_sets = [tuple((str(k), v) for k, v in p.items()) for p in param_grid]
    return SweepResult(
        param_sets=param_sets,
        fold_total_returns=fold_returns,
        train_window_bars=train_window,
        test_window_bars=test_window,
        n_folds=len(next(iter(fold_returns), [])),
        n_periods=n,
    )


def parameter_grid_around(
    baseline: ParameterSet,
    multipliers: Sequence[float] = (0.5, 1.0, 2.0),
) -> List[Dict[str, Any]]:
    """Build a parameter grid around a baseline by scaling each parameter.

    Every grid point scales every parameter by one multiplier, so the
    baseline itself is included (all multipliers at 1.0).

    Args:
        baseline: parameter values to scale, e.g. (("fast", 20), ("slow", 60)).
        multipliers: relative scaling factors per parameter; default gives
            half, the baseline, and double for every parameter.

    Returns:
        List of parameter dicts covering the Cartesian product.
    """
    baseline_dict = dict(baseline)
    grid: List[Dict[str, Any]] = []
    for combo in product(multipliers, repeat=len(baseline)):
        grid.append({str(k): baseline_dict[k] * float(m) for k, m in zip(baseline_dict, combo)})
    return grid


def random_signals(n: int, seed: Optional[int] = None) -> List[Signal]:
    """Coin-flip positioning, independent of prices.

    Used as the null hypothesis in robustness testing: a signal that
    never looks at the data. Any structure in its sweep results comes
    from the backtest framework rather than from the signal.
    """
    rng = np.random.default_rng(seed)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]


def noise_benchmark(
    bars,
    param_grid: List[Dict[str, Any]],
    train_window: int,
    test_window: int,
    warmup: int = 0,
    overlap_window: int = 0,
    cfg: BacktestConfig = BacktestConfig(),
    periods_per_year: int = 252,
    seed: Optional[int] = None,
) -> SweepSummary:
    """Run a parameter sweep on a coin-flip signal.

    Returns a SweepSummary whose medians should be near zero. Use it as
    the null hypothesis when judging whether a real signal's sweep shows
    structure beyond randomness.

    Args:
        bars: full bar series (length is used only for the signal shape).
        param_grid, train_window, test_window, warmup, overlap_window, cfg, periods_per_year:
            forwarded to parameter_sweep.
        seed: random seed for the coin flips.
    """
    n = len(bars)
    base_seed = seed if seed is not None else 42

    def _param_seed(params):
        # deterministic, process-independent mapping from params to a seed.
        total = sum(float(v) for v in params.values())
        return base_seed + int(round(total) * 1000)

    result = parameter_sweep(
        signals_fn=lambda closes, **params: random_signals(n, seed=_param_seed(params)),
        bars=bars,
        param_grid=param_grid,
        train_window=train_window,
        test_window=test_window,
        warmup=warmup,
        overlap_window=overlap_window,
        cfg=cfg,
        periods_per_year=periods_per_year,
    )
    # The baseline param set is the first set; its fold-level log returns are
    # the coin-flip null's fold distribution, used by ``compare_noise`` for
    # the sample-calibrated tolerance.
    fold_rets = np.array(result.fold_total_returns[0], dtype=np.float64)
    null_fold_log_returns = np.log1p(np.clip(fold_rets, -1.0 + 1e-12, None)).tolist()
    return sweep_summary(
        result,
        result.param_sets[0],
        noise_median=None,
        noise_fold_median_log_returns=null_fold_log_returns,
    )


def sweep_summary(
    result: SweepResult,
    baseline: ParameterSet,
    noise_median: Optional[float] = None,
    noise_fold_median_log_returns: Optional[Sequence[float]] = None,
) -> SweepSummary:
    """Summarize a SweepResult with respect to a baseline parameter set.

    Args:
        result: the sweep result.
        baseline: the reference parameter set; its median log return is the
            reference for the deviation scan and the noise comparison.
        noise_median: optional median log return from a noise benchmark;
            used to flag whether the baseline is indistinguishable from noise.
        noise_fold_median_log_returns: optional fold-level log returns of the
            noise benchmark's baseline param set (same ``n_folds``); used by
            ``compare_noise`` to compute a sample-calibrated tolerance.

    Returns:
        SweepSummary with a deviation scan, sign consistency, best/worst
        parameter sets, and the optional noise comparison.
    """
    fold_rets = np.array(result.fold_total_returns, dtype=np.float64)
    # equity blow-up (total_return <= -1) makes log1p undefined; clip to a
    # floor so the deviation scan stays finite and comparable across parameter
    # sets.
    log_rets = np.log1p(np.clip(fold_rets, -1.0 + 1e-12, None))
    medians = np.median(log_rets, axis=1).tolist()
    medians_array = np.array(medians, dtype=np.float64)

    baseline_dict = dict(baseline)
    dev_groups: Dict[float, List[float]] = {}
    for ps, med in zip(result.param_sets, medians):
        dev = round(max(abs(float(m) / baseline_dict[name] - 1.0) for name, m in ps), 6)
        dev_groups.setdefault(dev, []).append(med)
    sorted_devs = sorted(dev_groups)
    medians_at_deviation = {d: float(np.median(dev_groups[d])) for d in sorted_devs}

    n_positive = int(np.sum(medians_array > 0))
    n_negative = int(np.sum(medians_array < 0))
    n_zero = int(len(medians) - n_positive - n_negative)
    fraction_positive = n_positive / len(medians)

    best_idx = int(np.argmax(medians_array))
    worst_idx = int(np.argmin(medians_array))

    return SweepSummary(
        param_sets=result.param_sets,
        baseline_param_set=baseline,
        median_log_returns=medians,
        deviations=sorted_devs,
        medians_at_deviation=medians_at_deviation,
        n_param_sets=len(result.param_sets),
        n_positive=n_positive,
        n_zero=n_zero,
        n_negative=n_negative,
        fraction_positive=fraction_positive,
        best_param_set=result.param_sets[best_idx],
        best_median_log_return=medians[best_idx],
        worst_param_set=result.param_sets[worst_idx],
        worst_median_log_return=medians[worst_idx],
        noise_median_log_return=noise_median,
        n_folds=result.n_folds,
        n_periods=result.n_periods,
        train_window_bars=result.train_window_bars,
        test_window_bars=result.test_window_bars,
        noise_fold_median_log_returns=noise_fold_median_log_returns,
    )
