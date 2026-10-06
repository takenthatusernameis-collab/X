"""Regime-stability stress testing for walk-forward backtests.

A strategy that is robust to market conditions should not depend on one
particular regime configuration of the data. This module regenerates the
same series under a family of regime parameterizations (volatility,
transition speed, mean reversion, drift) and checks whether
out-of-sample walk-forward results behave consistently across them.

A robust strategy shows the same sign and roughly the same magnitude of
performance in every scenario. A fragile one shows a scenario-specific
edge (strong in one regime, noise-like in others); pure noise stays
indistinguishable from the noise benchmark in every scenario.

Usage::

    bars = generate_bars(2500, seed=42)

    grid = parameter_grid_around((("fast", 20), ("slow", 60)))
    result = regime_stress(
        signals_fn=ma_crossover_signals,
        bars=list(bars),
        param_grid=grid,
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
    )
    print(result.inspect())
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .data import Bar, BarSequence, Regime, generate_bars
from .engine import BacktestConfig, Signal
from .perturbation import (
    ParameterSet,
    noise_benchmark,
    parameter_grid_around,
    parameter_sweep,
    sweep_summary,
)

EdgeFree = 0
EdgePresent = 1
OUTLIER_TOL = 0.05
"""Tolerance in log-return units for judging whether a scenario's result
deviates from the noise benchmark (the framework's null hypothesis)."""

REGIME_STABLE_LOSS = "REGIME_STABLE_LOSS"
"""Verdict for a candidate whose results are consistently negative across
all scenarios: stable losses, not an edge. The medians are uniformly below
zero but the cross-scenario swing does not exceed twice the null
dispersion, i.e. the losses are regime-stable rather than regime-dependent.
This verdict must be read alongside the medians: it records stability of
losses, not a robust edge."""


@dataclass(frozen=True)
class RegimeScenario:
    """A parameterization of the data-generating process.

    Attributes:
        name: human-readable label.
        regimes: the regime mixture to generate the series under.
        p_transition: per-bar regime-switch probability.
        seed: deterministic seed for this scenario's series.
        start_price: starting price level.
    """
    name: str
    regimes: List[Regime]
    p_transition: float
    seed: int
    start_price: float = 100.0


def generate_under(scenario: RegimeScenario, n_bars: int) -> BarSequence:
    """Generate an `n_bars` series under this scenario's regime parameters."""
    return generate_bars(
        n_bars=n_bars,
        regimes=scenario.regimes,
        p_transition=scenario.p_transition,
        start_price=scenario.start_price,
        seed=scenario.seed,
    )


def conditional_signal(
    base_signals_fn: Callable[[NDArray, ...], List[Signal]],
    condition_fn: Callable[[float], bool],
) -> Callable[[NDArray, ...], List[Signal]]:
    """Wrap a signal function: emit its signals only when the early-vol
    condition holds.

    Used to construct contrived strategies that only "work" in one regime
    family, so the framework can detect regime-dependence.
    """

    def wrapper(closes: NDArray, **params: Any) -> List[Signal]:
        n = len(closes)
        window = min(60, max(4, n // 4))
        if n < window:
            return [Signal(date=i + 1, weight=0.0) for i in range(n)]
        early_vol = float(
            np.std(np.log(closes[window // 2 : window]), ddof=1)
        ) * np.sqrt(252.0)
        if not condition_fn(early_vol):
            return [Signal(date=i + 1, weight=0.0) for i in range(n)]
        return base_signals_fn(closes, **params)


    return wrapper


def active_when_turbulent(early_vol: float) -> bool:
    """Trade only in turbulent regimes (realized vol above 25% annualized)."""
    return early_vol > 0.25


def active_when_anything(_early_vol: float) -> bool:
    """Always emit signals — the control for a fully regime-independent edge."""
    return True


def deterministic_edge_signal(closes: NDArray, fast: int, slow: int) -> List[Signal]:
    """A simple mechanical signal with a fixed, regime-independent structure:

    long whenever the 20-bar mean exceeds a fixed level.

    This signal does not adapt to regimes; any edge it has is purely from
    the fixed rule, so it is an ideal contrived candidate for testing
    regime-dependence.
    """
    n = len(closes)
    out = [Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(19, n):
        if np.mean(closes[i - 19 : i + 1]) > 97.5:
            out[i] = Signal(date=i + 1, weight=1.0)
    return out


def direction_signal(closes: NDArray, fast: int, slow: int) -> List[Signal]:
    """A contrived signal that takes the direction of the opening move.

    If the first bar closes above the start price the series is an
    up-trend regime and the signal is long everywhere; otherwise it is
    short everywhere. The rule is past-only (it uses only bar 1) and is
    an ideal contrived candidate for testing regime-dependence.
    """
    n = len(closes)
    direction = 1.0 if closes[0] > 100.0 else -1.0
    return [Signal(date=i + 1, weight=direction) for i in range(n)]


def always_long_signal(closes: NDArray, fast: int, slow: int) -> List[Signal]:
    """A contrived signal that is long in every bar, regardless of price.

    It has no look-ahead and is purely mechanical; on a positive-drift
    regime it shows a large edge, on a negative-drift regime a large
    drag. It is an ideal contrived candidate for demonstrating
    regime-dependence.
    """
    n = len(closes)
    return [Signal(date=i + 1, weight=1.0) for i in range(n)]


@dataclass(frozen=True)
class RegimeScenarioResult:
    """Results for one regime scenario.

    The overall verdict of the surrounding regime-stability run is one of
    CONSISTENT_WITH_NOISE, REGIME_DEPENDENT, REGIME_STABLE, or
    REGIME_STABLE_LOSS (a candidate whose medians are uniformly negative
    across scenarios but whose cross-scenario swing stays within twice the
    null dispersion).
    """
    name: str
    param_sets: List[ParameterSet]
    baseline_param_set: ParameterSet
    baseline_median_log_return: float
    noise_median_log_return: float
    edge_status: int  # EdgeFree / EdgePresent
    n_folds: int
    n_periods: int
    median_log_returns: List[float]


def run_scenario(
    signals_fn: Callable[[NDArray, ...], List[Signal]],
    bars: BarSequence,
    param_grid: List[Dict[str, Any]],
    scenario: RegimeScenario,
    baseline: ParameterSet,
    train_window: int,
    test_window: int,
    warmup: int = 0,
    overlap_window: int = 0,
    cfg: BacktestConfig = BacktestConfig(),
    periods_per_year: int = 252,
) -> RegimeScenarioResult:
    """Run the full walk-forward perturbation sweep under one regime scenario.

    Args:
        signals_fn: past-only signal function, called as
            `signals_fn(closes, **params)`.
        bars: the bar series; its length determines signal shape.
        param_grid: parameter sets to sweep.
        scenario: the regime parameterization for this scenario's series.
        baseline: the reference parameter set.
        train_window, test_window, warmup, overlap_window, cfg, periods_per_year:
            forwarded to walk_forward.

    Returns:
        RegimeScenarioResult with the candidate's and the noise
        benchmark's medians for this scenario.
    """
    n_bars = len(bars)
    series = generate_under(scenario, n_bars)
    sweep = parameter_sweep(
        signals_fn=signals_fn,
        bars=series,
        param_grid=param_grid,
        train_window=train_window,
        test_window=test_window,
        warmup=warmup,
        overlap_window=overlap_window,
        cfg=cfg,
        periods_per_year=periods_per_year,
    )
    summary = sweep_summary(sweep, baseline)
    noise = noise_benchmark(
        series,
        param_grid=param_grid,
        train_window=train_window,
        test_window=test_window,
        warmup=warmup,
        overlap_window=overlap_window,
        cfg=cfg,
        periods_per_year=periods_per_year,
        seed=scenario.seed,
    )
    baseline_median = summary.baseline_median_log_return
    noise_median = noise.baseline_median_log_return

    edge_free = abs(baseline_median - noise_median) <= OUTLIER_TOL
    edge_status = EdgeFree if edge_free else EdgePresent

    return RegimeScenarioResult(
        name=scenario.name,
        param_sets=sweep.param_sets,
        baseline_param_set=baseline,
        baseline_median_log_return=baseline_median,
        noise_median_log_return=noise_median,
        edge_status=edge_status,
        n_folds=sweep.n_folds,
        n_periods=sweep.n_periods,
        median_log_returns=summary.median_log_returns,
    )


@dataclass(frozen=True)
class RegimeStressResult:
    """Results of a regime-stability stress test."""
    scenarios: List[RegimeScenarioResult]
    param_sets: List[ParameterSet]
    baseline_param_set: ParameterSet
    overall_verdict: str  # REGIME_STABLE / REGIME_STABLE_LOSS / REGIME_DEPENDENT / CONSISTENT_WITH_NOISE
    candidate_dispersion: float  # std of candidate medians across scenarios
    null_dispersion: float       # std of noise medians across scenarios
    n_folds: int
    n_periods: int
    train_window_bars: int
    test_window_bars: int

    @property
    def all_edge_free(self) -> bool:
        return all(abs(s.baseline_median_log_return) <= OUTLIER_TOL
                   for s in self.scenarios)

    @property
    def n_edge_free(self) -> int:
        return sum(1 for s in self.scenarios if abs(s.baseline_median_log_return) <= OUTLIER_TOL)

    @property
    def n_edge_present(self) -> int:
        return sum(1 for s in self.scenarios if abs(s.baseline_median_log_return) > OUTLIER_TOL)

    @property
    def regime_dependent(self) -> bool:
        return self.overall_verdict == "REGIME_DEPENDENT"

    def param_set_at(self, scenario_name: str, deviation: float) -> Optional[float]:
        """Median log return for a parameter set at a given deviation, for one
        scenario, if present."""
        s = next((sc for sc in self.scenarios if sc.name == scenario_name), None)
        if s is None:
            return None
        summary = sweep_summary(
            __dummy_result(s.param_sets, s.median_log_returns, s.n_folds), baseline
        )
        return summary.param_set_at(deviation)

    def inspect(self) -> str:
        lines = [
            "=== Regime-stability stress test ===",
            "",
            f"Scenarios: {[s.name for s in self.scenarios]}",
            f"Baseline: {self.baseline_param_set}",
            f"Folds per scenario: {self.n_folds}  Periods per scenario: {self.n_periods}",
            f"Train window: {self.train_window_bars}d  Test window: {self.test_window_bars}d",
            "",
            "Per-scenario results (median log return vs noise benchmark):",
        ]
        for s in self.scenarios:
            status = "EDGE  " if abs(s.baseline_median_log_return) > OUTLIER_TOL else "noise"
            lines.append(
                f"  {s.name:14s}: candidate {s.baseline_median_log_return:+.3f}, "
                f"noise {s.noise_median_log_return:+.3f}  [{status}]"
            )
        lines.append("")
        lines.append(
            f"Candidate dispersion across scenarios: {self.candidate_dispersion:+.3f}  "
            f"Null dispersion: {self.null_dispersion:+.3f}"
        )
        lines.append("")
        lines.append(f"Overall verdict: {self.overall_verdict}")
        lines.append("")
        lines.append(
            "Interpretation: REGIME_STABLE means the candidate behaves the same "
            "way in every regime mix (same edge, or no edge, everywhere). "
            "REGIME_STABLE_LOSS means the candidate loses money in every regime "
            "mix (stable losses, not an edge). REGIME_DEPENDENT means the "
            "candidate's results vary strongly across regimes relative to the "
            "noise benchmark (fits one regime, fails in others). "
            "CONSISTENT_WITH_NOISE means the candidate is "
            "indistinguishable from the coin-flip null in every scenario."
        )
        return "\n".join(lines)

    def compare_noise(self, scenario_name: str, tol: float = OUTLIER_TOL) -> bool:
        """Return True if the candidate's baseline is within `tol` of the
        noise benchmark for the given scenario."""
        s = next((sc for sc in self.scenarios if sc.name == scenario_name), None)
        if s is None:
            raise ValueError(f"unknown scenario {scenario_name}")
        return abs(s.baseline_median_log_return - s.noise_median_log_return) <= tol


# --- helpers to build a summary for inspect-only (for param_set_at) ----------


def __dummy_result(
    param_sets: List[ParameterSet],
    medians: List[float],
    n_folds: int,
) -> SweepResult:
    """Construct a minimal SweepResult so sweep_summary can address param sets.

    This is an internal helper used only for reporting; it does not rerun
    any backtests.
    """
    from .perturbation import SweepResult

    fold_returns = [[m] for m in medians]
    return SweepResult(
        param_sets=param_sets,
        fold_total_returns=fold_returns,
        train_window_bars=0,
        test_window_bars=0,
        n_folds=n_folds,
        n_periods=0,
    )


def canonical_regime_scenarios(base_seed: int = 42) -> List[RegimeScenario]:
    """Return the enterprise's canonical family of regime parameterizations,
    derived deterministically from a base seed.

    Covers a calm, a turbulent, a mean-reverting and a trending
    parameterization, spanning the regime space relevant to daily
    simulation.
    """
    return [
        RegimeScenario(
            name="calm",
            regimes=[
                Regime(drift_annual=0.03, vol_annual=0.12, intraday_range_scale=0.015),
                Regime(drift_annual=-0.01, vol_annual=0.18, intraday_range_scale=0.020),
            ],
            p_transition=0.005,
            seed=base_seed + 1,
        ),
        RegimeScenario(
            name="turbulent",
            regimes=[
                Regime(drift_annual=-0.02, vol_annual=0.40, intraday_range_scale=0.035),
                Regime(drift_annual=0.01, vol_annual=0.30, intraday_range_scale=0.025),
            ],
            p_transition=0.02,
            seed=base_seed + 7,
        ),
        RegimeScenario(
            name="mean_reverting",
            regimes=[
                Regime(
                    drift_annual=0.00,
                    vol_annual=0.25,
                    mean_reversion_speed=0.05,
                    mean_reversion_level=100.0,
                    intraday_range_scale=0.020,
                ),
            ],
            p_transition=0.00,
            seed=base_seed + 13,
        ),
        RegimeScenario(
            name="trending",
            regimes=[
                Regime(drift_annual=0.06, vol_annual=0.30, intraday_range_scale=0.025),
                Regime(drift_annual=0.02, vol_annual=0.22, intraday_range_scale=0.020),
            ],
            p_transition=0.01,
            seed=base_seed + 19,
        ),
    ]


def stress_regime_scenarios() -> List[RegimeScenario]:
    """Return an extreme regime family for demonstrating stress testing.

    The scenarios are deliberately severe (drifts of +-0.9/year at low
    volatility) so that regime dependence is easily measurable. They are
    not meant as realistic market parameterizations.
    """
    return [
        RegimeScenario(
            name="strong_up",
            regimes=[Regime(drift_annual=0.90, vol_annual=0.05)],
            p_transition=0.00,
            seed=999,
        ),
        RegimeScenario(
            name="neutral",
            regimes=[Regime(drift_annual=0.00, vol_annual=0.20)],
            p_transition=0.00,
            seed=998,
        ),
        RegimeScenario(
            name="strong_down",
            regimes=[Regime(drift_annual=-0.90, vol_annual=0.05)],
            p_transition=0.00,
            seed=997,
        ),
    ]


def regime_stress(
    signals_fn: Callable[[NDArray, ...], List[Signal]],
    bars: BarSequence,
    param_grid: List[Dict[str, Any]],
    baseline: ParameterSet,
    scenarios: Optional[List[RegimeScenario]] = None,
    train_window: int = 252,
    test_window: int = 84,
    warmup: int = 0,
    overlap_window: int = 0,
    cfg: BacktestConfig = BacktestConfig(),
    periods_per_year: int = 252,
) -> RegimeStressResult:
    """Stress-test a candidate signal across a family of regime scenarios.

    Args:
        signals_fn: past-only signal function, called as
            `signals_fn(closes, **params)`.
        bars: the bar series; only its length is used (a fresh series is
            generated per scenario).
        param_grid: parameter sets to sweep inside each scenario.
        baseline: the reference parameter set.
        scenarios: regime scenarios to run. If None, uses
            `canonical_regime_scenarios(len(bars))`.
        train_window, test_window, warmup, overlap_window, cfg, periods_per_year:
            forwarded to walk_forward.

    Returns:
        RegimeStressResult with per-scenario verdicts and an overall
        verdict.
    """
    n_bars = len(bars)
    scenario_list = scenarios if scenarios is not None else canonical_regime_scenarios(
        n_bars
    )
    scenario_results: List[RegimeScenarioResult] = []
    for scenario in scenario_list:
        res = run_scenario(
            signals_fn=signals_fn,
            bars=generate_under(scenario, n_bars),
            param_grid=param_grid,
            scenario=scenario,
            baseline=baseline,
            train_window=train_window,
            test_window=test_window,
            warmup=warmup,
            overlap_window=overlap_window,
            cfg=cfg,
            periods_per_year=periods_per_year,
        )
        scenario_results.append(res)

    candidate_medians = np.array([s.baseline_median_log_return for s in scenario_results])
    null_medians = np.array([s.noise_median_log_return for s in scenario_results])
    candidate_dispersion = float(np.std(candidate_medians))
    null_dispersion = float(np.std(null_medians))

    if not scenario_results:
        verdict = "CONSISTENT_WITH_NOISE"
    elif all(abs(s.baseline_median_log_return) <= OUTLIER_TOL for s in scenario_results):
        verdict = "CONSISTENT_WITH_NOISE"
    elif candidate_dispersion > 2.0 * null_dispersion:
        # The candidate's results swing across regimes far more than a
        # coin-flip null would: the candidate fits particular regime mixes
        # rather than being robust to the data-generating process.
        verdict = "REGIME_DEPENDENT"
    elif all(m < 0 for m in candidate_medians):
        # Every scenario shows a negative median but the cross-scenario swing
        # stays within twice the null dispersion: the candidate loses money
        # consistently across regimes. This is REGIME_STABLE_LOSS — stability
        # of losses, not a robust edge — and it must be read alongside the
        # medians rather than treated as an edge.
        verdict = REGIME_STABLE_LOSS
    else:
        verdict = "REGIME_STABLE"

    first = scenario_results[0]
    return RegimeStressResult(
        scenarios=scenario_results,
        param_sets=first.param_sets,
        baseline_param_set=first.baseline_param_set,
        overall_verdict=verdict,
        candidate_dispersion=candidate_dispersion,
        null_dispersion=null_dispersion,
        n_folds=first.n_folds,
        n_periods=first.n_periods,
        train_window_bars=train_window,
        test_window_bars=test_window,
    )


def volatility_segments(
    closes: NDArray,
    vol_calm: float = 0.15,
    vol_turbulent: float = 0.28,
    window: int = 60,
) -> List[str]:
    """Past-only classification of each bar by trailing realized volatility.

    Uses only closes[:i], so no look-ahead is possible. Bars return one of
    'calm', 'normal', or 'turbulent'; very short prefixes return 'insufficient'
    and are dropped as segments (they cannot hold a walk-forward).

    Args:
        closes: closing prices.
        vol_calm, vol_turbulent: annualized realized-vol thresholds.
        window: trailing-bar window for realized volatility.

    Returns:
        List of labels, one per bar.
    """
    labels: List[str] = []
    for i in range(len(closes)):
        if i < window:
            labels.append("insufficient")
        else:
            rv = float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(
                252.0
            )
            if rv < vol_calm:
                labels.append("calm")
            elif rv < vol_turbulent:
                labels.append("normal")
            else:
                labels.append("turbulent")
    return labels


def segment_fn_from_labels(labels: List[str]) -> Callable[[NDArray, int], str]:
    """Turn a precomputed past-only label list into the segment_fn callable
    expected by ``stress_segments``. Labels do not depend on closes, so the
    closes argument is unused."""

    def fn(_closes: NDArray, i: int) -> str:
        return labels[i]

    return fn


def stress_segments(
    signals_fn: Callable[[NDArray, ...], List[Signal]],
    bars: BarSequence,
    signals: List[Signal],
    param_grid: List[Dict[str, Any]],
    baseline: ParameterSet,
    segment_fn: Callable[[NDArray, int], str],
    train_window: int,
    test_window: int,
    warmup: int = 0,
    overlap_window: int = 0,
    cfg: BacktestConfig = BacktestConfig(),
    periods_per_year: int = 252,
    min_segment_bars: int = 400,
) -> RegimeStressResult:
    """Stress-test a candidate across contiguous segments of a real series
    classified by a past-only regime function.

    Each bar is classified by `segment_fn(closes, i)` using only closes[:i]
    (past-only), producing contiguous segments. Each qualifying segment is
    backtested with walk-forward IS/OOS, and the candidate's per-segment
    median log returns are compared against a coin-flip null benchmark run on
    the same segments, using the same verdict logic as ``regime_stress``:
    CONSISTENT_WITH_NOISE (no edge in any segment), REGIME_DEPENDENT
    (candidate's cross-segment swing exceeds 2x the null's), or
    REGIME_STABLE (consistent edge or no edge everywhere).

    Segments are shorter than a full series by construction, so walk-forward
    fold counts may differ across segments; per-segment fold counts are
    reported in the result.

    Args:
        signals_fn: past-only signal function, called as
            `signals_fn(closes, **params)`.
        bars: the bar series (full length).
        signals: the full past-only signal series (same length as `bars`).
        param_grid: parameter sets to sweep inside each segment.
        baseline: the reference parameter set.
        segment_fn: callable(closes: np.ndarray, i: int) -> segment label,
            using closes[:i] only (past-only).
        train_window, test_window, warmup, overlap_window, cfg, periods_per_year:
            forwarded to walk_forward.
        min_segment_bars: minimum segment length to include; shorter segments
            cannot hold a meaningful walk-forward.

    Returns:
        RegimeStressResult with one scenario per segment and an overall
        verdict.
    """
    if isinstance(bars, BarSequence):
        closes = bars.closes_array()
    else:
        closes = np.array([float(b.close) for b in bars], dtype=np.float64)

    # BarSequence only supports single-index getitem, so convert once here.
    bars_list = list(bars) if isinstance(bars, BarSequence) else bars

    if len(signals) != len(closes):
        raise ValueError("signals must have the same length as bars")

    labels = [segment_fn(closes, i) for i in range(len(closes))]
    segments: List[Tuple[str, int, int]] = []
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - start >= min_segment_bars:
                segments.append((cur, start, i))
            cur = labels[i]
            start = i
    if len(labels) - start >= min_segment_bars:
        segments.append((cur, start, len(labels)))

    if not segments:
        raise ValueError(f"no segment has >= {min_segment_bars} bars")

    scenario_results: List[RegimeScenarioResult] = []
    import inspect
    for name, s, e in segments:
        seg_bars = bars_list[s:e]
        seg_signals = signals[s:e]

        # Wrap the signal function so that it routes by the same labels that
        # define the segments. For signal functions that classify their input
        # series on their own (e.g. a regime-adaptive MA), recomputing labels on
        # a segment slice would differ from the global classification used to
        # find the segment; passing the segment's label slice makes the routing
        # consistent with the segment boundaries. Only functions exposing a
        # `labels` parameter receive it; the others are passed through unchanged.
        _sig = inspect.signature(signals_fn)
        if "labels" in _sig.parameters:
            segment_signals_fn = (
                lambda closes, **params: signals_fn(closes, labels=labels[s:e], **params)
            )
        else:
            segment_signals_fn = signals_fn

        sweep = parameter_sweep(
            signals_fn=segment_signals_fn,
            bars=seg_bars,
            param_grid=param_grid,
            train_window=train_window,
            test_window=test_window,
            warmup=warmup,
            overlap_window=overlap_window,
            cfg=cfg,
            periods_per_year=periods_per_year,
        )
        cand_summary = sweep_summary(sweep, baseline)
        noise = noise_benchmark(
            seg_bars,
            param_grid=param_grid,
            train_window=train_window,
            test_window=test_window,
            warmup=0,
            overlap_window=0,
            cfg=cfg,
            periods_per_year=periods_per_year,
        )
        edge_status = (
            EdgeFree
            if abs(cand_summary.baseline_median_log_return - noise.baseline_median_log_return)
            <= OUTLIER_TOL
            else EdgePresent
        )
        scenario_results.append(RegimeScenarioResult(
            name=name,
            param_sets=sweep.param_sets,
            baseline_param_set=baseline,
            baseline_median_log_return=cand_summary.baseline_median_log_return,
            noise_median_log_return=noise.baseline_median_log_return,
            edge_status=edge_status,
            n_folds=sweep.n_folds,
            n_periods=sweep.n_periods,
            median_log_returns=cand_summary.median_log_returns,
        ))

    candidate_medians = np.array([s.baseline_median_log_return for s in scenario_results])
    null_medians = np.array([s.noise_median_log_return for s in scenario_results])
    candidate_dispersion = float(np.std(candidate_medians))
    null_dispersion = float(np.std(null_medians))

    if all(abs(s.baseline_median_log_return) <= OUTLIER_TOL for s in scenario_results):
        verdict = "CONSISTENT_WITH_NOISE"
    elif candidate_dispersion > 2.0 * null_dispersion:
        verdict = "REGIME_DEPENDENT"
    elif all(m < 0 for m in candidate_medians):
        # Every segment shows a negative median but the cross-segment swing
        # stays within twice the null dispersion: the candidate loses money
        # consistently across regimes. This is REGIME_STABLE_LOSS — stability
        # of losses, not a robust edge — and it must be read alongside the
        # medians rather than treated as an edge.
        verdict = REGIME_STABLE_LOSS
    else:
        verdict = "REGIME_STABLE"

    first = scenario_results[0]
    return RegimeStressResult(
        scenarios=scenario_results,
        param_sets=first.param_sets,
        baseline_param_set=first.baseline_param_set,
        overall_verdict=verdict,
        candidate_dispersion=candidate_dispersion,
        null_dispersion=null_dispersion,
        n_folds=first.n_folds,
        n_periods=first.n_periods,
        train_window_bars=train_window,
        test_window_bars=test_window,
    )


# ==============================================================================
# Reusable real-data utilities
# ==============================================================================

def volatility_blocks(
    closes: NDArray,
    n_blocks: int = 4,
    window: int = 60,
) -> List[str]:
    """Split a real series into `n_blocks` contiguous blocks by date and label
    each block by its realized-volatility regime.

    Each block is labeled 'turbulent' if its median trailing-`window`-bar
    realized volatility (annualized) exceeds the series-wide median, else
    'calm'. Bars before `window` bars into the series are 'insufficient' and
    dropped as a short leading segment (they cannot hold a walk-forward).

    Labels are determined by past data only: the block medians and the
    series-wide median are both computed from closes. The labels do not
    depend on the bar index `i`, so the series is classified once, before
    any backtest runs.

    Args:
        closes: closing prices (full length).
        n_blocks: number of contiguous blocks to split the series into.
        window: trailing-bar window for realized volatility.

    Returns:
        List of labels, one per bar ('calm' / 'turbulent' / 'insufficient').
    """
    n = len(closes)
    block_size = n // n_blocks
    bar_vols: List[float] = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0)
            )
    active = [v for v in bar_vols if v > 0]
    if not active:
        raise ValueError("not enough bars to compute realized volatility")
    series_med = float(np.median(active))

    labels: List[str] = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            block = i // block_size
            s, e = block * block_size, (block + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window
                else 0.0
            )
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def ma_crossover_signals(
    closes: NDArray,
    fast: int = 20,
    slow: int = 60,
) -> List[Signal]:
    """Past-only dual moving-average crossover signal (long/short).

    Long when the fast MA > slow MA, short otherwise; neutral before the
    slow window. This matches the signals used in `examples/regime_stability_*`
    so regime-stability runs on the collected universe use the same signal
    contract as the AAPL/NVDA runs.

    No look-ahead: each signal uses only closes up to and including its own
    bar.
    """
    n = len(closes)
    out: List[Signal] = [Signal(date=i + 1, weight=0.0) for i in range(n)]
    fast, slow = int(fast), int(slow)
    for i in range(max(fast, slow) - 1, n):
        fast_ma = float(np.mean(closes[i - fast + 1 : i + 1]))
        slow_ma = float(np.mean(closes[i - slow + 1 : i + 1]))
        out[i] = Signal(date=i + 1, weight=1.0 if fast_ma > slow_ma else -1.0)
    return out


def mean_reversion_signals(
    closes: NDArray,
    lookback: int = 5,
) -> List[Signal]:
    """Past-only short-horizon return-reversal signal (new signal class).

    Short the previous ``lookback``-day return and hold for one day: if the
    last ``lookback`` days were up, go short; if they were down, go long.
    Neutral before the ``lookback`` window. Daily rebalancing (a fresh
    signal at every bar) — the standard 1-day-ahead reversal implementation.

    This is the enterprise's new-signal-class test candidate: mean
    reversion on large-cap US equities, the conceptual opposite of the
    trend-following MA crossover tested previously. Like the MA crossover,
    it is judged through the same perturbation / coin-flip-null /
    regime-stability gates before any positive claim is considered.

    No look-ahead: each signal uses only closes up to and including its own
    bar.
    """
    n = len(closes)
    out: List[Signal] = [Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = Signal(date=i + 1, weight=-1.0 if ret > 0 else 1.0)
    return out


def momentum_signals(
    closes: NDArray,
    lookback: int = 5,
) -> List[Signal]:
    """Past-only short-horizon momentum signal (new signal class).

    Long the previous ``lookback``-day return and hold for one day: if the
    last ``lookback`` days were up, go long; if they were down, go short.
    Neutral before the ``lookback`` window. Daily rebalancing (a fresh
    signal at every bar) — the standard 1-day-ahead momentum implementation.

    This is the conceptual opposite of `mean_reversion_signals` (same
    construction, opposite sign). It is the competing hypothesis to the
    reversal class tested previously: if shorting the previous N-day return
    produces uniformly negative results, the momentum variant should
    produce uniformly positive results; either outcome settles whether the
    5-day-lookback direction bet is a robust edge on this universe.

    No look-ahead: each signal uses only closes up to and including its own
    bar.
    """
    n = len(closes)
    out: List[Signal] = [Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def breakout_signals(
    closes: NDArray,
    lookback: int = 20,
) -> List[Signal]:
    """Past-only breakout-continuation signal (new signal class).

    Long the previous ``lookback``-day breakout and hold for one day: a
    breakout occurs when the current close exceeds the highest close of the
    preceding ``lookback`` bars; at a breakout, go long for one day. Neutral
    before the ``lookback`` window and on non-breakout bars. Daily rebalancing
    (a fresh signal at every bar). Bars before the ``lookback`` window are
    neutral.

    Unlike ``momentum_signals``, which bets on the sign of the
    ``lookback``-day return, the breakout filter requires the price to make a
    new ``lookback``-day high before taking a position — a directionally
    biased, threshold-activated momentum rule. This is the enterprise's
    new-signal-class test candidate: 20-day breakout continuation on large-cap
    US equities. It is judged through the same perturbation / coin-flip-null /
    regime-stability gates before any positive claim is considered.

    No look-ahead: each signal uses only closes up to and including its own
    bar.
    """
    n = len(closes)
    out: List[Signal] = [Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        if float(closes[i]) > float(np.max(closes[i - lookback : i])):
            out[i] = Signal(date=i + 1, weight=1.0)
    return out


def regime_adaptive_ma_signals(
    closes: NDArray,
    fast_calm: int = 20,
    slow_calm: int = 60,
    fast_turb: int = 10,
    slow_turb: int = 30,
    window: int = 60,
    labels: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[Signal]:
    """Past-only MA-crossover whose windows adapt to the volatility regime.

    In 'calm' segments use the (fast_calm, slow_calm) windows; in 'turbulent'
    segments use the shorter (fast_turb, slow_turb) windows to react faster to
    quicker moves. Regime labels are computed from ``volatility_blocks``
    (past-only, from closes) when ``labels`` is not supplied, or taken from the
    caller-supplied ``labels`` list (one per bar of ``closes``) — this second
    form lets ``stress_segments`` route signals by the same labels that define
    the segments instead of re-classifying each segment slice on its own. Bars
    before ``window`` are 'insufficient' and emit a neutral signal.

    This is a principled, fixed a-priori design, not tuned to OOS results. The
    test it serves: the base MA(20/60) is REGIME_DEPENDENT on AMZN and JPM
    because their edge lived in the 2009-2013 turbulent block. If that edge is
    genuinely volatility-regime-contingent and harvestable, a regime-adaptive
    implementation should pass the regime-stability gate (REGIME_STABLE). If it
    stays REGIME_DEPENDENT (or becomes CONSISTENT_WITH_NOISE), the edge is
    period-specific and cannot be separated by a volatility-regime classifier.
    """
    if labels is None:
        labels = volatility_blocks(closes, n_blocks=4, window=window)
    n = len(closes)
    signals: List[Signal] = [Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(n):
        lab = labels[i]
        if lab == "insufficient":
            continue
        fast, slow = (fast_turb, slow_turb) if lab == "turbulent" else (fast_calm, slow_calm)
        if i >= slow - 1:
            fm = float(np.mean(closes[i - fast + 1 : i + 1]))
            sm = float(np.mean(closes[i - slow + 1 : i + 1]))
            signals[i] = Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
    return signals


# ==============================================================================
# Universe-wide regime stability
# ==============================================================================

@dataclass(frozen=True)
class RegimeUniverseSummary:
    """Aggregate regime-stability results across a universe of tickers.

    Attributes:
        assets: per-asset `RegimeStressResult`, keyed by ticker in the order
            the tickers were loaded.
        ticker_order: ticker order matching ``assets``.
    """
    assets: Dict[str, RegimeStressResult]
    ticker_order: List[str]

    @property
    def n_assets(self) -> int:
        """Number of tickers evaluated."""
        return len(self.assets)

    @property
    def verdict_counts(self) -> Dict[str, int]:
        """Counts of each overall verdict across assets."""
        counts: Dict[str, int] = {
            "CONSISTENT_WITH_NOISE": 0,
            "REGIME_STABLE": 0,
            "REGIME_STABLE_LOSS": 0,
            "REGIME_DEPENDENT": 0,
        }
        for res in self.assets.values():
            counts[res.overall_verdict] += 1
        return counts

    def inspect(self) -> str:
        lines = [
            "=== Regime-stability across universe ===",
            "",
            f"Tickers: {len(self.ticker_order)}  (order: {', '.join(self.ticker_order)})",
            f"Verdict counts: {', '.join(f'{k}={v}' for k, v in self.verdict_counts.items())}",
            "",
            "Per-asset regime-stability results:",
        ]
        for ticker in self.ticker_order:
            res = self.assets[ticker]
            segs = "[" + ", ".join(s.name for s in res.scenarios) + "]"
            medians = "[" + ", ".join(
                f"{s.baseline_median_log_return:+.3f}" for s in res.scenarios
            ) + "]"
            lines.append(
                f"  {ticker:6s}: segments={segs} candidate_medians={medians} "
                f"dispersion=+{res.candidate_dispersion:.3f} "
                f"verdict={res.overall_verdict}"
            )
        lines.append("")
        lines.append(
            "Interpretation: REGIME_STABLE = the candidate behaves the same way "
            "(same edge or no edge) across regimes; REGIME_STABLE_LOSS = the "
            "candidate loses money in every regime mix (stable losses, not an "
            "edge, and the verdict must be read alongside the medians); "
            "REGIME_DEPENDENT = results swing strongly across regimes relative "
            "to the coin-flip null (fits one regime); CONSISTENT_WITH_NOISE = "
            "indistinguishable from noise in every segment."
        )
        return "\n".join(lines)


def stress_segments_across_tickers(
    tickers: Dict[str, BarSequence],
    signals_fn: Callable[[NDArray, ...], List[Signal]],
    regime_labels_fn: Callable[[NDArray], List[str]],
    baseline: ParameterSet = (("fast", 20), ("slow", 60)),
    param_grid: List[Dict[str, Any]] = [{"fast": 20, "slow": 60}],
    train_window: int = 252,
    test_window: int = 84,
    warmup: int = 60,
    overlap_window: int = 0,
    cfg: BacktestConfig = BacktestConfig(),
    periods_per_year: int = 252,
    min_segment_bars: int = 400,
) -> RegimeUniverseSummary:
    """Run regime-stability stress testing across a universe of tickers.

    Each ticker is classified into contiguous volatility regime segments
    (past-only, via ``regime_labels_fn``), then ``stress_segments`` is run
    on each asset with the same candidate signal, parameter set, and
    walk-forward settings. Per-asset verdicts are aggregated into a
    ``RegimeUniverseSummary`` so the cross-asset pattern of regime-stability
    verdicts can be judged (e.g. whether one asset's REGIME_STABLE verdict
    generalizes, or whether a REGIME_DEPENDENT signal is flagged consistently
    across assets).

    This is the reusable real-data regime family for regime-stability: it
    replaces the ad-hoc per-ticker examples with a single deterministic
    call whose settings are fixed in one place, making universe-level
    regime-stability reproducible and auditable.

    Args:
        tickers: dict ticker -> BarSequence of adjusted closes (same source
            for all assets so segments are comparable).
        signals_fn: past-only signal function, called as
            `signals_fn(closes, **params)`.
        regime_labels_fn: callable(closes) -> List[str] of labels, one per bar,
            computed from closes only (past-only).
        baseline: reference parameter set.
        param_grid: parameter sets swept inside each segment.
        train_window, test_window, warmup, overlap_window, cfg, periods_per_year:
            forwarded to walk_forward.
        min_segment_bars: minimum segment length to include; shorter segments
            cannot hold a meaningful walk-forward.

    Returns:
        RegimeUniverseSummary with per-asset verdicts and a count summary.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty dict")

    ticker_order = list(tickers.keys())
    assets: Dict[str, RegimeStressResult] = {}
    for ticker in ticker_order:
        bars = tickers[ticker]
        closes = bars.closes_array() if isinstance(bars, BarSequence) else np.array(
            [float(b.close) for b in bars], dtype=np.float64
        )
        labels = regime_labels_fn(closes)
        seg_fn = segment_fn_from_labels(labels)
        signals = signals_fn(closes, **{k: float(v) for k, v in dict(baseline).items()})
        if len(signals) != len(closes):
            raise ValueError(
                f"signals length ({len(signals)}) != bars length ({len(closes)}) "
                f"for {ticker}"
            )
        assets[ticker] = stress_segments(
            signals_fn=signals_fn,
            bars=bars,
            signals=signals,
            param_grid=param_grid,
            baseline=baseline,
            segment_fn=seg_fn,
            train_window=train_window,
            test_window=test_window,
            warmup=warmup,
            overlap_window=overlap_window,
            cfg=cfg,
            periods_per_year=periods_per_year,
            min_segment_bars=min_segment_bars,
        )
    return RegimeUniverseSummary(assets=assets, ticker_order=ticker_order)


# ==============================================================================
# Cross-sectional relative strength (CSRS) signal class helpers
# ==============================================================================

def csrs_spread_daily_returns(tickers, lookback=20, top_k=3, bottom_k=3):
    """Cross-sectional relative-strength spread: daily P&L of a long/short
    portfolio built from ranks across the collected universe.

    At each date t >= lookback: compute each ticker's lookback return
    (closes[i-lookback : i+1], past-only); rank tickers descending; long the
    top_k, short the bottom_k with equal capital per leg; hold one day;
    daily rebalance. The spread P&L at bar t equals the mean of the
    long-legs' next-day simple returns minus the mean of the short-legs'
    next-day simple returns. Bars before lookback, and days with fewer than
    2*k tickers available, are neutral (P&L = 0).

    No look-ahead: every input at bar t uses closes only up to and
    including t; the realized P&L is the return realized between bar t
    and bar t+1.

    Args:
        tickers: dict ticker -> BarSequence of adjusted closes (all tickers
            share the same date grid).
        lookback: lookback window in bars.
        top_k, bottom_k: leg sizes.

    Returns:
        np.ndarray of daily spread P&L (simple returns), one entry per bar
        (0.0 where no trade is possible).
    """
    min_len = min(len(tb) for tb in tickers.values())
    daily_pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(lookback, min_len - 1):
        # truncate all series to the fully overlapping window so the spread
        # is computed on the same date grid for every ticker; each ticker
        # contributes a lookback return only where it has both the lookback
        # window and the next-day close (past-only).
        closes = {t: tb.closes_array()[:min_len] for t, tb in tickers.items()}
        rets = {t: float(closes[t][i] / closes[t][i - lookback] - 1)
                for t in closes}
        sorted_tickers = sorted(rets, key=lambda t: rets[t], reverse=True)
        longs = sorted_tickers[:top_k]
        shorts = sorted_tickers[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        long_ret = float(np.mean(
            [closes[t][i + 1] / closes[t][i] - 1 for t in longs]))
        short_ret = float(np.mean(
            [closes[t][i + 1] / closes[t][i] - 1 for t in shorts]))
        daily_pnl[i] = long_ret - short_ret
    return daily_pnl


def csrs_null_spread_family(family, lookback, top_k, bottom_k, seed=42):
    """Coin-flip null of the CSRS spread over a synthetic family of assets.

    Ranks are computed identically (past-only) across the family; the
    long/short assignment is randomized by a deterministic per-bar coin flip,
    preserving the spread's magnitude/structure while destroying its
    information content.

    Args:
        family: list of BarSequence, one per asset (same length).
        lookback, top_k, bottom_k: as in csrs_spread_family.
        seed: deterministic RNG seed for the coin flips.

    Returns:
        np.ndarray of daily P&L (one entry per bar).
    """
    rng = np.random.default_rng(seed)
    first_len = len(family[0])
    daily_pnl = np.zeros(first_len, dtype=np.float64)
    for i in range(lookback, first_len - 1):
        rets = {}
        for a, asset in enumerate(family):
            c = asset.closes_array()
            rets[a] = float(np.log(c[i]) - np.log(c[i - lookback]))
        sorted_assets = sorted(rets, key=lambda a: rets[a], reverse=True)
        longs = sorted_assets[:top_k]
        shorts = sorted_assets[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        leg_ret = lambda a: float(np.log(family[a].closes_array()[i + 1])
                                  - np.log(family[a].closes_array()[i]))
        spread = np.mean([leg_ret(a) for a in longs]) - np.mean([leg_ret(a) for a in shorts])
        daily_pnl[i] = spread if rng.random() < 0.5 else -spread
    return daily_pnl


def csrs_null_spread_daily_returns(tickers, lookback=20, top_k=3, bottom_k=3,
                                   seed=42):
    """Coin-flip null of the CSRS spread.

    Ranks are computed identically (past-only); the long/short assignment is
    randomized: a deterministic coin flip per bar decides whether the
    rank-ordered legs are held long (as in csrs_spread_daily_returns) or
    shorted. This preserves the spread's magnitude/structure while removing
    its information content.

    Args:
        tickers: dict ticker -> BarSequence.
        lookback, top_k, bottom_k: as in csrs_spread_daily_returns.
        seed: deterministic RNG seed for the coin flips.

    Returns:
        np.ndarray of daily P&L (one entry per bar).
    """
    rng = np.random.default_rng(seed)
    min_len = min(len(tb) for tb in tickers.values())
    daily_pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(lookback, min_len - 1):
        closes = {t: tb.closes_array()[:min_len] for t, tb in tickers.items()}
        rets = {t: float(closes[t][i] / closes[t][i - lookback] - 1)
                for t in closes}
        sorted_tickers = sorted(rets, key=lambda t: rets[t], reverse=True)
        longs = sorted_tickers[:top_k]
        shorts = sorted_tickers[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        leg_ret = lambda t: float(closes[t][i + 1] / closes[t][i] - 1)
        spread = np.mean([leg_ret(t) for t in longs]) - np.mean([leg_ret(t) for t in shorts])
        daily_pnl[i] = spread if rng.random() < 0.5 else -spread
    return daily_pnl


# ==============================================================================
# Cross-sectional momentum signal class helpers
# ==============================================================================

def cs_momentum_spread_daily_returns(tickers, lookback=63, top_k=3, bottom_k=3):
    """Cross-sectional momentum spread: daily P&L of a rank-based portfolio
    built from the cross section of collected tickers.

    At each date t >= lookback: compute each ticker's lookback return
    (closes[i-lookback : i+1], past-only); rank tickers descending; long the
    top_k, short the bottom_k with equal capital per leg; hold one day;
    daily rebalance. The spread P&L at bar t equals the mean of the
    long-legs' next-day simple returns minus the mean of the short-legs'
    next-day simple returns. Bars before lookback, and days with fewer than
    2*k tickers available, are neutral (P&L = 0).

    This is the cross-sectional (portfolio-level) analog of the momentum
    signal class (go long the previous N-day return), extended from a
    per-asset bet to a rank-based cross-sectional spread. It is
    conceptually distinct from the cross-sectional relative-strength class:
    it uses the longer, classical momentum lookback (63 bars = 13 weeks)
    rather than 20. No look-ahead: every input at bar t uses closes only up
    to and including t; the realized P&L is the return realized between bar t
    and bar t+1.

    Args:
        tickers: dict ticker -> BarSequence of adjusted closes (all tickers
            share the same date grid).
        lookback: lookback window in bars for the cumulative return.
        top_k, bottom_k: leg sizes.

    Returns:
        np.ndarray of daily spread P&L (simple returns), one entry per bar
        (0.0 where no trade is possible).
    """
    min_len = min(len(tb) for tb in tickers.values())
    daily_pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(lookback, min_len - 1):
        # truncate all series to the fully overlapping window so the spread
        # is computed on the same date grid for every ticker; each ticker
        # contributes a lookback return only where it has both the lookback
        # window and the next-day close (past-only).
        closes = {t: tb.closes_array()[:min_len] for t, tb in tickers.items()}
        rets = {t: float(closes[t][i] / closes[t][i - lookback] - 1)
                for t in closes}
        sorted_tickers = sorted(rets, key=lambda t: rets[t], reverse=True)
        longs = sorted_tickers[:top_k]
        shorts = sorted_tickers[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        long_ret = float(np.mean(
            [closes[t][i + 1] / closes[t][i] - 1 for t in longs]))
        short_ret = float(np.mean(
            [closes[t][i + 1] / closes[t][i] - 1 for t in shorts]))
        daily_pnl[i] = long_ret - short_ret
    return daily_pnl


def cs_momentum_null_spread_daily_returns(tickers, lookback=63, top_k=3,
                                          bottom_k=3, seed=42):
    """Coin-flip null of the cross-sectional momentum spread.

    Ranks are computed identically (past-only); the long/short assignment is
    randomized: a deterministic coin flip per bar decides whether the
    rank-ordered legs are held long (as in cs_momentum_spread_daily_returns)
    or shorted. This preserves the spread's magnitude/structure while removing
    its information content.

    Args:
        tickers: dict ticker -> BarSequence.
        lookback, top_k, bottom_k: as in cs_momentum_spread_daily_returns.
        seed: deterministic RNG seed for the coin flips.

    Returns:
        np.ndarray of daily P&L (one entry per bar).
    """
    rng = np.random.default_rng(seed)
    min_len = min(len(tb) for tb in tickers.values())
    daily_pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(lookback, min_len - 1):
        closes = {t: tb.closes_array()[:min_len] for t, tb in tickers.items()}
        rets = {t: float(closes[t][i] / closes[t][i - lookback] - 1)
                for t in closes}
        sorted_tickers = sorted(rets, key=lambda t: rets[t], reverse=True)
        longs = sorted_tickers[:top_k]
        shorts = sorted_tickers[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        leg_ret = lambda t: float(closes[t][i + 1] / closes[t][i] - 1)
        spread = np.mean([leg_ret(t) for t in longs]) - np.mean([leg_ret(t) for t in shorts])
        daily_pnl[i] = spread if rng.random() < 0.5 else -spread
    return daily_pnl


def cs_momentum_spread_family(family, lookback, top_k, bottom_k):
    """Compute the cross-sectional momentum spread daily P&L across a
    synthetic family of BarSequence assets (same pattern as
    cs_momentum_spread_daily_returns but over a family instead of a dict of
    tickers).

    Args:
        family: list of BarSequence, one per asset (same length).
        lookback, top_k, bottom_k: as in cs_momentum_spread_daily_returns.

    Returns:
        np.ndarray of daily spread P&L.
    """
    first_len = len(family[0])
    daily_pnl = np.zeros(first_len, dtype=np.float64)
    for i in range(lookback, first_len - 1):
        rets = {}
        for a, asset in enumerate(family):
            c = asset.closes_array()
            rets[a] = float(np.log(c[i]) - np.log(c[i - lookback]))
        sorted_assets = sorted(rets, key=lambda a: rets[a], reverse=True)
        longs = sorted_assets[:top_k]
        shorts = sorted_assets[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        leg_ret = lambda a: float(np.log(family[a].closes_array()[i + 1])
                                  - np.log(family[a].closes_array()[i]))
        long_ret = float(np.mean([leg_ret(a) for a in longs]))
        short_ret = float(np.mean([leg_ret(a) for a in shorts]))
        daily_pnl[i] = long_ret - short_ret
    return daily_pnl


def cs_momentum_null_spread_family(family, lookback, top_k, bottom_k,
                                   seed=42):
    """Coin-flip null of the cross-sectional momentum spread over a synthetic
    family of assets.

    Ranks are computed identically (past-only); the long/short assignment is
    randomized by a deterministic per-bar coin flip, preserving the spread's
    magnitude/structure while destroying its information content.

    Args:
        family: list of BarSequence, one per asset (same length).
        lookback, top_k, bottom_k: as in cs_momentum_spread_family.
        seed: deterministic RNG seed for the coin flips.

    Returns:
        np.ndarray of daily P&L (one entry per bar).
    """
    rng = np.random.default_rng(seed)
    first_len = len(family[0])
    daily_pnl = np.zeros(first_len, dtype=np.float64)
    for i in range(lookback, first_len - 1):
        rets = {}
        for a, asset in enumerate(family):
            c = asset.closes_array()
            rets[a] = float(np.log(c[i]) - np.log(c[i - lookback]))
        sorted_assets = sorted(rets, key=lambda a: rets[a], reverse=True)
        longs = sorted_assets[:top_k]
        shorts = sorted_assets[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        leg_ret = lambda a: float(np.log(family[a].closes_array()[i + 1])
                                  - np.log(family[a].closes_array()[i]))
        spread = np.mean([leg_ret(a) for a in longs]) \
                 - np.mean([leg_ret(a) for a in shorts])
        daily_pnl[i] = spread if rng.random() < 0.5 else -spread
    return daily_pnl


def build_synthetic_spread_asset(daily_pnl, start_price=100.0):
    """Build a BarSequence whose close path equals the cumulative equity of a
    spread P&L series, with a constant-long signal suitable for the engine.

    The synthetic asset carries the spread's compounded equity path; when it
    is backtested fully invested (signal weight 1.0 at every bar), the
    framework's walk-forward OOS total_return equals the spread's compounded
    return over the OOS test window:

        total_return = prod(1 + daily_pnl[OOS_bars]) - 1

    so the walk-forward machinery can be reused for the spread. Neutral bars
    carry the equity forward (daily_pnl = 0 before lookback is harmless).

    Args:
        daily_pnl: np.ndarray of daily spread P&L (one entry per bar).
        start_price: starting equity level of the synthetic asset.

    Returns:
        BarSequence with close == cumulative equity and open/high/low derived
        from the day-to-day move (range within the day's high/low; no
        look-ahead).
    """
    n = len(daily_pnl)
    eq = np.empty(n, dtype=np.float64)
    eq[0] = start_price * (1.0 + daily_pnl[0]) if daily_pnl[0] != 0.0 else start_price
    for i in range(1, n):
        r = daily_pnl[i]
        eq[i] = eq[i - 1] * (1.0 + r) if r != 0.0 else eq[i - 1]

    opens = np.empty(n, dtype=np.float64)
    opens[0] = start_price
    opens[1:] = eq[:-1]
    lows = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    lows[0] = highs[0] = eq[0]
    lows[1:] = np.minimum(eq[:-1], eq[1:])
    highs[1:] = np.maximum(eq[:-1], eq[1:])
    volumes = np.full(n, 1_000_000.0, dtype=np.float64)
    dates = np.arange(1, n + 1, dtype=np.int64)
    return BarSequence(dates, opens, highs, lows, eq, volumes)


def csrs_spread_family(family, lookback, top_k, bottom_k):
    """Compute the CSRS spread daily P&L across a synthetic family of
    BarSequence assets (same pattern as csrs_spread_daily_returns but over a
    family instead of a dict of tickers).

    Args:
        family: list of BarSequence, one per asset (same length).
        lookback, top_k, bottom_k: as in csrs_spread_daily_returns.

    Returns:
        np.ndarray of daily spread P&L.
    """
    first_len = len(family[0])
    daily_pnl = np.zeros(first_len, dtype=np.float64)
    for i in range(lookback, first_len - 1):
        rets = {}
        for a, asset in enumerate(family):
            c = asset.closes_array()
            rets[a] = float(np.log(c[i]) - np.log(c[i - lookback]))
        sorted_assets = sorted(rets, key=lambda a: rets[a], reverse=True)
        longs = sorted_assets[:top_k]
        shorts = sorted_assets[-bottom_k:]
        if len(longs) < top_k or len(shorts) < bottom_k:
            continue
        leg_ret = lambda a: float(np.log(family[a].closes_array()[i + 1])
                                  - np.log(family[a].closes_array()[i]))
        long_ret = float(np.mean([leg_ret(a) for a in longs]))
        short_ret = float(np.mean([leg_ret(a) for a in shorts]))
        daily_pnl[i] = long_ret - short_ret
    return daily_pnl


# ==============================================================================
# Volatility-targeting signal class helpers
# ==============================================================================

def vol_rank_spread_daily_returns(tickers, lookback=60, top_k=3, bottom_k=3):
    """Volatility-targeting spread: daily P&L of a long-low-vol / short-high-vol
    portfolio built from ranks across the collected universe.

    At each date t >= lookback: compute each ticker's trailing-`lookback`-day
    realized volatility (closes[i-lookback : i], past-only); rank tickers
    descending by volatility; long the bottom_k (lowest-volatility) tickers,
    short the top_k (highest-volatility) tickers with equal capital per leg;
    hold one day; daily rebalance. The spread P&L at bar t equals the mean of
    the long-legs' next-day simple returns minus the mean of the short-legs'
    next-day simple returns. Bars before lookback, and days with fewer than
    2*k tickers available, are neutral (P&L = 0).

    This is the volatility-targeting / low-vol premium signal class on the
    collected universe: rank by past volatility, long bottom-k / short top-k,
    hold 1 day. Past-only by construction.

    No look-ahead: every input at bar t uses closes only up to and
    including t; the realized P&L is the return realized between bar t
    and bar t+1.

    Args:
        tickers: dict ticker -> BarSequence of adjusted closes (all tickers
            share the same date grid).
        lookback: lookback window in bars for realized volatility.
        top_k, bottom_k: leg sizes.

    Returns:
        np.ndarray of daily spread P&L (simple returns), one entry per bar
        (0.0 where no trade is possible).
    """
    min_len = min(len(tb) for tb in tickers.values())
    daily_pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(lookback, min_len - 1):
        closes = {t: tb.closes_array()[:min_len] for t, tb in tickers.items()}
        vols = {t: float(np.std(np.log(closes[t][i - lookback : i]), ddof=1))
                for t in closes}
        sorted_tickers = sorted(vols, key=lambda t: vols[t], reverse=True)
        longs = sorted_tickers[-bottom_k:]   # lowest-volatility leg
        shorts = sorted_tickers[:top_k]      # highest-volatility leg
        if len(longs) < bottom_k or len(shorts) < top_k:
            continue
        long_ret = float(np.mean(
            [closes[t][i + 1] / closes[t][i] - 1 for t in longs]))
        short_ret = float(np.mean(
            [closes[t][i + 1] / closes[t][i] - 1 for t in shorts]))
        daily_pnl[i] = long_ret - short_ret
    return daily_pnl


def vol_rank_spread_family(family, lookback, top_k, bottom_k):
    """Compute the volatility-targeting spread daily P&L across a synthetic
    family of BarSequence assets (same pattern as vol_rank_spread_daily_returns
    but over a family instead of a dict of tickers).

    Args:
        family: list of BarSequence, one per asset (same length).
        lookback, top_k, bottom_k: as in vol_rank_spread_daily_returns.

    Returns:
        np.ndarray of daily spread P&L.
    """
    first_len = len(family[0])
    daily_pnl = np.zeros(first_len, dtype=np.float64)
    for i in range(lookback, first_len - 1):
        vols = {}
        for a, asset in enumerate(family):
            c = asset.closes_array()
            vols[a] = float(np.std(c[i - lookback : i], ddof=1))
        sorted_assets = sorted(vols, key=lambda a: vols[a], reverse=True)
        longs = sorted_assets[-bottom_k:]
        shorts = sorted_assets[:top_k]
        if len(longs) < bottom_k or len(shorts) < top_k:
            continue
        leg_ret = lambda a: float(np.log(family[a].closes_array()[i + 1])
                                  - np.log(family[a].closes_array()[i]))
        long_ret = float(np.mean([leg_ret(a) for a in longs]))
        short_ret = float(np.mean([leg_ret(a) for a in shorts]))
        daily_pnl[i] = long_ret - short_ret
    return daily_pnl


def vol_rank_null_spread_daily_returns(tickers, lookback=60, top_k=3, bottom_k=3,
                                       seed=42):
    """Coin-flip null of the volatility-targeting spread.

    Ranks are computed identically (past-only); the long/short assignment is
    randomized: a deterministic coin flip per bar decides whether the
    rank-ordered legs are held (as in vol_rank_spread_daily_returns) or
    shorted. This preserves the spread's magnitude/structure while removing
    its information content.

    Args:
        tickers: dict ticker -> BarSequence.
        lookback, top_k, bottom_k: as in vol_rank_spread_daily_returns.
        seed: deterministic RNG seed for the coin flips.

    Returns:
        np.ndarray of daily P&L (one entry per bar).
    """
    rng = np.random.default_rng(seed)
    min_len = min(len(tb) for tb in tickers.values())
    daily_pnl = np.zeros(min_len, dtype=np.float64)
    for i in range(lookback, min_len - 1):
        closes = {t: tb.closes_array()[:min_len] for t, tb in tickers.items()}
        vols = {t: float(np.std(np.log(closes[t][i - lookback : i]), ddof=1))
                for t in closes}
        sorted_tickers = sorted(vols, key=lambda t: vols[t], reverse=True)
        longs = sorted_tickers[-bottom_k:]
        shorts = sorted_tickers[:top_k]
        if len(longs) < bottom_k or len(shorts) < top_k:
            continue
        leg_ret = lambda t: float(closes[t][i + 1] / closes[t][i] - 1)
        spread = float(np.mean([leg_ret(t) for t in longs])) \
                 - float(np.mean([leg_ret(t) for t in shorts]))
        daily_pnl[i] = spread if rng.random() < 0.5 else -spread
    return daily_pnl


def vol_rank_null_spread_family(family, lookback, top_k, bottom_k, seed=42):
    """Coin-flip null of the volatility-targeting spread over a synthetic
    family of assets.

    Ranks are computed identically (past-only); the long/short assignment is
    randomized by a deterministic per-bar coin flip, preserving the spread's
    magnitude/structure while destroying its information content.

    Args:
        family: list of BarSequence, one per asset (same length).
        lookback, top_k, bottom_k: as in vol_rank_spread_family.
        seed: deterministic RNG seed for the coin flips.

    Returns:
        np.ndarray of daily P&L (one entry per bar).
    """
    rng = np.random.default_rng(seed)
    first_len = len(family[0])
    daily_pnl = np.zeros(first_len, dtype=np.float64)
    for i in range(lookback, first_len - 1):
        vols = {}
        for a, asset in enumerate(family):
            c = asset.closes_array()
            vols[a] = float(np.std(c[i - lookback : i], ddof=1))
        sorted_assets = sorted(vols, key=lambda a: vols[a], reverse=True)
        longs = sorted_assets[-bottom_k:]
        shorts = sorted_assets[:top_k]
        if len(longs) < bottom_k or len(shorts) < top_k:
            continue
        leg_ret = lambda a: float(np.log(family[a].closes_array()[i + 1])
                                  - np.log(family[a].closes_array()[i]))
        spread = np.mean([leg_ret(a) for a in longs]) \
                 - np.mean([leg_ret(a) for a in shorts])
        daily_pnl[i] = spread if rng.random() < 0.5 else -spread
    return daily_pnl
