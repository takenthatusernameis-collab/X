"""Asset-universe robustness testing for walk-forward backtests.

A strategy that is robust to the data-generating process should not depend on
one particular asset. This module runs the same walk-forward validation across
a family of assets and summarizes whether out-of-sample performance is
consistent across assets or concentrated in a few of them.

This closes the last robustness dimension in the methodology: perturbation
(parameter sensitivity), regime stability (same asset, different regimes),
and now asset-universe stability (same signal, different assets). The
anti-gaming discipline ("Do not treat a single fold, asset, regime, toy
dataset, or example as evidence of general profitability") is enforced
operationally by the concentration/consistency verdict.

Usage::

    assets = bt.uniform_regime_assets(
        base_regimes,
        n_assets=7,
        drift_offsets=[-0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12],
        seed=42,
    )
    grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
    result = bt.sweep_across_assets(
        signals_fn=ma_crossover_signals,
        assets=assets,
        param_grid=grid,
        baseline=(("fast", 20), ("slow", 60)),
        train_window=252,
        test_window=84,
        warmup=60,
        overlap_window=60,
    )
    summary = bt.asset_sweep_summary(result, baseline=(("fast", 20), ("slow", 60)))
    summary.inspect()

Research/simulation only. Not live trading.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .data import Bar, BarSequence, Regime, generate_bars
from .engine import BacktestConfig, Signal, walk_forward
from .perturbation import (
    ParameterSet,
    noise_benchmark,
    parameter_grid_around,
)

ASSET_VERDICT_CONCENTRATED = "CONCENTRATED"
ASSET_VERDICT_CONSISTENT = "CONSISTENT"
ASSET_VERDICT_NO_EDGE = "NO_EDGE"
CONCENTRATION_THRESHOLD = 0.60
"""A single asset's share of all positive-fold wins above this value flags the
sweep as concentrated (edge lives in a few assets rather than generalizing)."""


@dataclass(frozen=True)
class AssetSweepResult:
    """Results of one asset-universe sweep."""

    assets: List[BarSequence]
    asset_names: List[str]
    param_sets: List[ParameterSet]
    fold_total_returns: List[List[List[float]]]
    # [asset][param_set][fold]
    baseline_median_log_returns: List[float]  # per asset, same order as asset names
    n_folds: int
    n_periods: int
    train_window_bars: int
    test_window_bars: int

    @property
    def n_assets(self) -> int:
        return len(self.asset_names)

    @property
    def param_set_median_log_returns(self) -> List[List[float]]:
        """Per asset, per parameter set: median log return across folds.

        Derived from `fold_total_returns`; the walk-forwards are already done,
        so this adds no further computation.
        """
        out: List[List[float]] = []
        for asset_returns in self.fold_total_returns:
            log_rets = np.log1p(np.clip(
                np.array(asset_returns, dtype=np.float64), -1.0 + 1e-12, None))
            out.append(np.median(log_rets, axis=1).tolist())
        return out


@dataclass(frozen=True)
class AssetSweepSummary:
    """Summarized asset-universe robustness output."""

    asset_names: List[str]
    baseline_param_set: ParameterSet
    median_log_returns: List[float]  # per asset, same order as asset names
    n_assets: int
    positive_assets: int
    n_folds: int
    n_periods: int
    train_window_bars: int
    test_window_bars: int
    best_asset: str
    best_median_log_return: float
    worst_asset: str
    worst_median_log_return: float
    assets_positive_share: float  # best asset's share of the positive edge
    best_asset_concentration: float  # best asset's median share of positive edge pool
    null_median_log_return: Optional[float]
    null_dispersion: float
    asset_null_medians: Optional[List[float]]
    """Coin-flip baseline median per asset when per_asset_null=True, else
    None. Each entry is the coin-flip signal's median log return on the same
    asset, so the comparison matches each asset's own noise level."""
    asset_significance: Optional[List[bool]]
    """Per-asset significance flag: the asset's candidate median lies outside
    max(0.05, 2*null_dispersion) of its own coin-flip null baseline."""
    n_significant_assets: int
    """Count of assets with a significant edge under the per-asset flag. Equal
    to positive_assets when per_asset_null=False."""
    perturbation_profiles: Dict[str, List[Tuple[float, float]]]
    """Per-asset perturbation profiles: (deviation from baseline, median log
    return) sorted by ascending deviation. One profile per asset over the full
    sweep grid."""
    baseline_peak_count: int
    """Number of assets for which the baseline parameter set is the best
    (argmax) of its parameter-set medians."""
    baseline_peak_share: float
    """Fraction of assets whose baseline parameter set is the best of its
    parameter-set medians (baseline_peak_count / n_assets)."""
    positive_param_sets_per_asset: List[int]
    """For each asset, the number of parameter sets (of the grid) whose median
    log return across folds is positive."""
    baseline_in_grid: bool
    """Whether the baseline parameter set was present in the sweep grid.
    Baseline peak statistics are only defined in that case."""

    @property
    def verdict(self) -> str:
        """CONSISTENT / CONCENTRATED / NO_EDGE."""
        if self.n_assets == 0:
            return ASSET_VERDICT_NO_EDGE
        effective_tol = max(0.05, 2.0 * self.null_dispersion)
        # No edge anywhere: every asset's median lies within tolerance of its
        # own coin-flip null (when a per-asset null is available), otherwise of
        # the global null benchmark. Matching each asset to its own null keeps
        # the check faithful to that asset's noise level.
        if self.asset_null_medians is not None:
            within_tol = sum(
                1 for m, n in zip(self.median_log_returns, self.asset_null_medians)
                if abs(m - n) <= effective_tol
            )
        else:
            within_tol = sum(
                1 for m in self.median_log_returns
                if self.null_median_log_return is not None
                and abs(m - self.null_median_log_return) <= effective_tol
            )
        if within_tol >= self.n_assets:
            return ASSET_VERDICT_NO_EDGE
        # Concentrated: the best asset accounts for the majority of wins.
        if self.assets_positive_share >= CONCENTRATION_THRESHOLD:
            return ASSET_VERDICT_CONCENTRATED
        # Consistent: a majority of assets carry an edge and no single asset
        # dominates the win count.
        return ASSET_VERDICT_CONSISTENT

    @property
    def fraction_positive(self) -> float:
        return self.positive_assets / self.n_assets if self.n_assets else 0.0

    def inspect(self) -> str:
        lines = [
            "=== Asset-universe robustness sweep ===",
            "",
            f"Assets: {self.n_assets}  Baseline: {self.baseline_param_set}",
            f"Folds per asset: {self.n_folds}  Periods per asset: {self.n_periods}",
            f"Train window: {self.train_window_bars}d  Test window: {self.test_window_bars}d",
            "",
            "Median log return per asset:",
        ]
        for name, med in zip(self.asset_names, self.median_log_returns):
            lines.append(f"  {name:18s}: {med:+.3f}")
        lines.append("")
        lines.append(
            f"Median log return positive / negative across assets: "
            f"{self.positive_assets} / {self.n_assets - self.positive_assets}"
        )
        lines.append(
            f"Best asset: {self.best_asset} ({self.best_median_log_return:+.3f}); "
            f"worst asset: {self.worst_asset} ({self.worst_median_log_return:+.3f})"
        )
        if self.asset_significance is not None:
            effective_tol = max(0.05, 2.0 * self.null_dispersion)
            lines.append("")
            lines.append(
                f"Per-asset significance vs own coin-flip null (tolerance "
                f"{effective_tol:+.3f}):"
            )
            flag_lines = [
                f"  {name}: {m:+.3f} vs own-null {n:+.3f} -> "
                + ("edge  " if sig else "noise")
                for name, m, n, sig in zip(
                    self.asset_names,
                    self.median_log_returns,
                    self.asset_null_medians,
                    self.asset_significance,
                )
            ]
            lines.extend(flag_lines)
            lines.append(
                f"Significant assets: {self.n_significant_assets} / {self.n_assets}"
            )
        lines.append(
            f"Best asset's share of the positive edge: "
            f"{self.assets_positive_share:.1%}"
        )
        if self.null_median_log_return is not None:
            lines.append("")
            lines.append(
                f"Null (coin-flip across assets): median log return "
                f"{self.null_median_log_return:+.3f}, null dispersion {self.null_dispersion:+.3f}"
            )
        lines.append("")
        lines.append("Per-asset perturbation profiles (median log return by parameter")
        lines.append("set, ascending deviation from the baseline):")
        for name, pairs in self.perturbation_profiles.items():
            lines.append(f"  {name}:")
            for dev, med in pairs:
                marker = "  <-- baseline" if abs(dev) < 1e-12 else ""
                lines.append(f"    deviation {dev:4.2f}x:  {med:+.3f}{marker}")
        lines.append("")
        baseline_peak = self.baseline_peak_count
        if self.baseline_in_grid:
            lines.append(
                f"Baseline parameter set is the best for {baseline_peak} / {self.n_assets} assets"
                f" (baseline peak share {self.baseline_peak_share:.1%})."
            )
            for name, c in zip(self.asset_names, self.positive_param_sets_per_asset):
                lines.append(f"  {name}: {c} of {len(self.perturbation_profiles[name])} parameter sets positive")
        else:
            for name, c in zip(self.asset_names, self.positive_param_sets_per_asset):
                lines.append(f"  {name}: {c} of {len(self.perturbation_profiles[name])} parameter sets positive")
            lines.append("  (baseline parameter set was not present in the sweep grid; "
                         "baseline peak statistics are not defined).")
        lines.append("")
        lines.append(f"Overall verdict: {self.verdict}")
        lines.append("")
        lines.append(
            "Interpretation: CONSISTENT means the strategy earns its edge in a "
            "majority of assets without one asset dominating the wins. "
            "CONCENTRATED means the wins cluster in a few assets (best asset "
            "carries >= 60% of the positive edge); that is not evidence of "
            "general profitability. NO_EDGE means the strategy is "
            "indistinguishable from the coin-flip null in every asset."
        )
        return "\n".join(lines)


def sweep_across_assets(
    signals_fn: Callable[[NDArray, ...], List[Signal]],
    assets: Sequence[BarSequence],
    param_grid: List[Dict[str, Any]],
    baseline: ParameterSet,
    train_window: int,
    test_window: int,
    warmup: int = 0,
    overlap_window: int = 0,
    cfg: BacktestConfig = BacktestConfig(),
    periods_per_year: int = 252,
    asset_names: Optional[Sequence[str]] = None,
) -> AssetSweepResult:
    """Run walk-forward backtests for every parameter set across every asset.

    Args:
        signals_fn: callable(closes: np.ndarray, **params) returning a list
            of Signal with the same length as `bars`. Must be past-only: it
            may use only closes up to and including each signal bar's date
            and must not touch open/high/low.
        assets: iterable of BarSequence, one per asset in the universe.
        param_grid: list of parameter sets; each is a dict of name -> value.
        baseline: the reference parameter set for the summary.
        train_window, test_window, warmup, overlap_window, cfg, periods_per_year:
            forwarded to walk_forward.
        asset_names: optional human-readable names, one per asset. Default:
            "asset_0", "asset_1", ...

    Returns:
        AssetSweepResult with per-asset, per-parameter-set fold returns.
    """
    assets = list(assets)
    if not assets:
        raise ValueError("assets must contain at least one BarSequence")

    names = [str(n) for n in asset_names] if asset_names else None
    if names is None:
        names = [f"asset_{i}" for i in range(len(assets))]
    if len(names) != len(assets):
        raise ValueError("asset_names length must match number of assets")

    param_sets = [tuple((str(k), v) for k, v in p.items()) for p in param_grid]
    fold_returns: List[List[List[float]]] = []
    baseline_medians: List[float] = []

    cfg = replace(cfg, periods_per_year=periods_per_year)

    for asset in assets:
        closes = asset.closes_array()
        n = len(closes)
        asset_returns: List[List[float]] = []
        for params in param_grid:
            signals = signals_fn(closes, **params)
            if len(signals) != n:
                raise ValueError(
                    f"signals_fn returned {len(signals)} signals for "
                    f"{n} bars in {names[len(fold_returns)]}"
                )
            result = walk_forward(
                bars=list(asset),
                signals=signals,
                train_window=train_window,
                test_window=test_window,
                warmup=warmup,
                overlap_window=overlap_window,
                cfg=cfg,
            )
            asset_returns.append([float(f.metrics["total_return"])
                                  for f in result.folds])
        fold_returns.append(asset_returns)
        fold_rets = np.array(asset_returns, dtype=np.float64)
        log_rets = np.log1p(np.clip(fold_rets, -1.0 + 1e-12, None))
        baseline_medians.append(float(np.median(log_rets)))

    return AssetSweepResult(
        assets=assets,
        asset_names=names,
        param_sets=param_sets,
        fold_total_returns=fold_returns,
        baseline_median_log_returns=baseline_medians,
        n_folds=len(next(iter(fold_returns), [])),
        n_periods=len(next(iter(assets), [])),
        train_window_bars=train_window,
        test_window_bars=test_window,
    )


def asset_sweep_summary(
    result: AssetSweepResult,
    baseline: ParameterSet,
    noise_n: int = 3,
    cfg: Optional[BacktestConfig] = None,
    per_asset_null: bool = True,
) -> AssetSweepSummary:
    """Summarize an AssetSweepResult with concentration and verdict logic.

    Args:
        result: the asset sweep result.
        baseline: the reference parameter set.
        noise_n: number of coin-flip assets run as a null benchmark to
            estimate framework noise dispersion (ignored when
            per_asset_null=True).
        cfg: backtest config used for the null benchmark; defaults to the
            configuration used in the sweep (BacktestConfig()).
        per_asset_null: default True. Run the coin-flip null separately for
            each asset and report per-asset significance flags against each
            asset's own null baseline. The NO_EDGE check then matches every
            asset to its own null instead of a single global null. When
            False, only the first `noise_n` assets are used, which preserves
            the pre-2026-10-03 behavior.

    Returns:
        AssetSweepSummary with a verdict of CONSISTENT / CONCENTRATED /
        NO_EDGE, plus per-asset medians, concentration measures, and
        (when per_asset_null=False — the prior API) per-asset significance
        flags set to None.
    """
    cfg = cfg or BacktestConfig()
    medians = np.array(result.baseline_median_log_returns, dtype=np.float64)
    best_idx = int(np.argmax(medians))
    worst_idx = int(np.argmin(medians))
    positive_count = int(np.sum(medians > 0))

    # Concentration: how much of the realized edge (sum of positive medians
    # across assets) is carried by the single best asset.
    positive_medians = [m for m in medians if m > 0.0]
    if positive_medians:
        total_positive_median = float(sum(positive_medians))
        best_positive_median = float(np.max(medians))
        assets_positive_share = best_positive_median / total_positive_median
    else:
        assets_positive_share = 0.0

    # Null benchmark: run coin-flip signals to estimate framework noise.
    # When per_asset_null, the null is computed separately for each asset so
    # the NO_EDGE check and the per-asset significance flags match every asset
    # to its own null baseline; otherwise only the first `noise_n` assets are
    # used, which preserves the prior behavior.
    grid = parameter_grid_around(baseline)
    n_assets = len(result.assets)
    if per_asset_null:
        n_null_assets = n_assets
    else:
        n_null_assets = min(noise_n, n_assets)

    null_medians_per_asset: List[float] = []
    for a in range(n_null_assets):
        noise = noise_benchmark(
            result.assets[a],
            param_grid=grid,
            train_window=result.train_window_bars,
            test_window=result.test_window_bars,
            warmup=0,
            overlap_window=0,
            cfg=cfg,
            periods_per_year=252,
            seed=42 + a * 1000,
        )
        null_medians_per_asset.append(noise.baseline_median_log_return)

    # Global null reference for the summary: mean of the computed null medians.
    null_median = (
        float(np.mean(null_medians_per_asset)) if null_medians_per_asset else 0.0
    )
    null_dispersion = (
        float(np.std(null_medians_per_asset, ddof=1))
        if len(null_medians_per_asset) >= 2
        else 0.0
    )

    # Per-asset significance: compare each asset's candidate median against
    # its own null baseline; None when per_asset_null=False to preserve the
    # prior API.
    if per_asset_null:
        asset_null_medians = null_medians_per_asset
        effective_tol = max(0.05, 2.0 * null_dispersion)
        asset_significance = [
            bool(abs(m - n) > effective_tol)
            for m, n in zip(medians, asset_null_medians)
        ]
        n_significant_assets = int(np.sum(asset_significance))
    else:
        asset_null_medians = None
        asset_significance = None
        n_significant_assets = positive_count

    # Per-asset perturbation profiles: median log return for each parameter
    # set in the sweep grid, expressed as deviation from the baseline.
    # The walk-forwards for the grid are already done
    # (AssetSweepResult.fold_total_returns), so this adds no further
    # computation.
    baseline_dict = dict(baseline)

    # Robust normalization of a parameter set to a list of (name, value)
    # pairs. Handles both the tuple-of-tuples form (("fast", 20), ("slow", 60))
    # used by sweep_across_assets and the flat ("fast", 20) pair form used in
    # some tests.
    def _param_pairs(ps: Any) -> List[Tuple[str, Any]]:
        if not ps:
            return []
        if isinstance(ps[0], tuple) and len(ps[0]) >= 2:
            return [(str(p[0]), p[1]) for p in ps]
        return [(str(a), b) for a, b in zip(ps[0::2], ps[1::2])]

    baseline_idx = next(
        (i for i, ps in enumerate(result.param_sets)
         if _param_pairs(ps) == _param_pairs(baseline)),
        None,
    )
    param_set_medians = result.param_set_median_log_returns
    perturbation_profiles: Dict[str, List[Tuple[float, float]]] = {}
    baseline_peak_count = 0
    positive_param_sets_per_asset: List[int] = []
    for name, med in zip(result.asset_names, param_set_medians):
        medians_array = np.array(med, dtype=np.float64)
        devs = [
            round(max(abs(v / baseline_dict[k] - 1.0) for k, v in _param_pairs(ps)), 6)
            for ps in result.param_sets
        ]
        perturbation_profiles[name] = sorted(zip(devs, medians_array.tolist()))
        if baseline_idx is not None:
            baseline_med = medians_array[baseline_idx]
            # baseline is the best parameter set for this asset when its median
            # equals the maximum across parameter sets (tie-safe).
            if baseline_med >= medians_array.max() - 1e-12:
                baseline_peak_count += 1
        positive_param_sets_per_asset.append(int(np.sum(medians_array > 0)))

    n_assets = len(result.asset_names)
    baseline_in_grid = baseline_idx is not None
    baseline_peak_share = (
        baseline_peak_count / n_assets if n_assets else 0.0
    )

    return AssetSweepSummary(
        asset_names=result.asset_names,
        baseline_param_set=baseline,
        median_log_returns=medians.tolist(),
        n_assets=len(result.asset_names),
        positive_assets=positive_count,
        n_folds=result.n_folds,
        n_periods=result.n_periods,
        train_window_bars=result.train_window_bars,
        test_window_bars=result.test_window_bars,
        best_asset=result.asset_names[best_idx],
        best_median_log_return=float(medians[best_idx]),
        worst_asset=result.asset_names[worst_idx],
        worst_median_log_return=float(medians[worst_idx]),
        assets_positive_share=assets_positive_share,
        best_asset_concentration=assets_positive_share,
        null_median_log_return=null_median,
        null_dispersion=null_dispersion,
        asset_null_medians=asset_null_medians,
        asset_significance=asset_significance,
        n_significant_assets=n_significant_assets,
        perturbation_profiles=perturbation_profiles,
        baseline_peak_count=baseline_peak_count,
        baseline_peak_share=baseline_peak_share,
        positive_param_sets_per_asset=positive_param_sets_per_asset,
        baseline_in_grid=baseline_in_grid,
    )


def uniform_regime_assets(
    base_regimes: Sequence[Regime],
    n_assets: int,
    drift_offsets: Sequence[float],
    seed: int = 42,
    n_bars: int = 1000,
    start_price: float = 100.0,
) -> List[BarSequence]:
    """Deterministic family of assets with a shared regime shape but shifted
    drifts.

    The regime mix (volatility, transitions, intraday range) is common across
    the family so that any systematic across-asset pattern is driven by the
    drift offsets, not by changes in the regime structure. This produces a
    realistic, controllable universe for robustness testing: a strategy with a
    genuine drift/trend edge should improve monotonically with the offset,
    while a regime-dependent strategy will be concentrated in the highest-
    offset assets.

    Args:
        base_regimes: the regime family shared by all assets.
        n_assets: number of assets in the universe.
        drift_offsets: annualized drift added to every regime of each asset;
            length must equal n_assets.
        seed: deterministic seed.
        n_bars: length of each asset's series.
        start_price: starting price level for each asset.

    Returns:
        List of n_assets BarSequences.
    """
    if len(drift_offsets) != n_assets:
        raise ValueError("drift_offsets length must equal n_assets")

    assets: List[BarSequence] = []
    for i, offset in enumerate(drift_offsets):
        regimes = [
            Regime(
                drift_annual=r.drift_annual + offset,
                vol_annual=r.vol_annual,
                mean_reversion_speed=r.mean_reversion_speed,
                mean_reversion_level=r.mean_reversion_level,
                p_transition=r.p_transition,
                intraday_range_scale=r.intraday_range_scale,
            )
            for r in base_regimes
        ]
        assets.append(
            generate_bars(
                n_bars=n_bars,
                regimes=regimes,
                start_price=start_price,
                seed=seed + i * 1000,
            )
        )
    return assets


def flat_regime_assets(n_assets: int, seed: int = 42, n_bars: int = 1000,
                       start_price: float = 100.0) -> List[BarSequence]:
    """Assets with near-zero drift and mean reversion — a no-edge reference
    universe.

    Suitable as a negative control: any strategy run on this universe should
    show medians near zero in every asset, i.e. the sweep reports NO_EDGE.
    """
    base = [
        Regime(drift_annual=0.0, vol_annual=0.20, mean_reversion_speed=0.02,
               mean_reversion_level=start_price, intraday_range_scale=0.02),
        Regime(drift_annual=-0.01, vol_annual=0.30, mean_reversion_speed=0.02,
               mean_reversion_level=start_price, intraday_range_scale=0.03),
    ]
    return uniform_regime_assets(base, n_assets, [0.0] * n_assets, seed=seed,
                                 n_bars=n_bars, start_price=start_price)
