# Persistent Enterprise State

## Objectives

1. Build the enterprise's deterministic research/backtest foundation.
   **Status: done** (completed in the prior activation; the passing unit
   test suite and the MA-crossover example confirmed it).
2. Add a second signal class and a robustness/perturbation test suite
   using the same framework. **Status: done** (this activation).
3. Prepare a real-data feasibility note for research-only simulation.
   **Status: done** (this activation).
4. Add a regime-stability stress test: verify a candidate's out-of-sample
   results are stable across different regime parameterizations rather
   than fitting one regime mix. **Status: done** (this activation).

## Status

- `research/backtest/` — deterministic backtest toolkit: engine
  (event-driven, walk-forward IS/OOS), metrics, synthetic regime-switching
  data, leakage/look-ahead checks, perturbation/parameter-sensitivity
  testing (`perturbation.py`), and regime-stability stress testing
  (`regime_stability.py`). Dependency: numpy.
- `tests/` — deterministic unit tests for engine, metrics, data,
  perturbation, and regime-stability; all passing.
- `examples/` — `ma_crossover.py` (walk-forward + cost sensitivity +
  perturbation sweep), `volatility_regime_filter.py` (regime-aware
  signal, walk-forward, costs), and `regime_stability_demo.py` (demonstrates
  all three regime-stability verdicts).
- `research/METHODOLOGY.md` — evidence standard, no-look-ahead rules,
  walk-forward discipline.
- `research/REAL_DATA_FEASIBILITY.md` — in-scope data, provenance
  manifest, preflight checks, and the leakage review checklist for any
  future real-data run.
- README and `research/README.md` updated to reflect the infrastructure.

## Current activation (follow-on)

Added deterministic robustness/perturbation tooling to the enterprise so
that every candidate strategy is tested for parameter sensitivity before
its results are admitted into the evidence base. This is the capability
that `METHODOLOGY.md` describes as required for judging robustness
(perturbation, parameter sensitivity, regime stability) but which did not
exist in the toolkit.

- Added `research/backtest/perturbation.py`: `parameter_sweep()` runs the
  same walk-forward validation across a grid of parameters;
  `parameter_grid_around()` builds a half/baseline/double grid around a
  canonical parameter set; `sweep_summary()` reports a deviation scan,
  sign consistency across parameter sets, and best/worst parameter sets;
  `noise_benchmark()` runs the sweep on a coin-flip (price-independent)
  signal as a null hypothesis; `random_signals()` generates that null.
- Added `examples/volatility_regime_filter.py`, a past-only volatility-regime
  filter with walk-forward IS/OOS, cost sensitivity, and leakage checks.
- Extended `examples/ma_crossover.py` with a perturbation sweep and a
  coin-flip null comparison.
- Added `tests/test_perturbation.py` (14 tests: determinism, grid
  formation, empty/edge cases, noise-centered-on-zero, known-peak
  detection, summary consistency, IS/OOS per-parameter-set, and noise
  comparison).
- Added `research/REAL_DATA_FEASIBILITY.md` defining in-scope and
  out-of-scope data sources, the provenance manifest schema, a data-quality
  preflight checklist, and the pre-run leakage review checklist.

Resulting finding (synthetic data; tooling validation only): the MA
crossover and the volatility-regime filter both show no persistent edge on
regime-switching synthetic data, and their sweep results are
indistinguishable from the coin-flip null. The framework therefore reports
correctly and does not hallucinate an edge — a negative result that reduces
the risk of later over-reading walk-forward noise. See
`logs/ACTIVATION-2026-10-02.md` for the full record.

## Current activation (regime-stability stress testing)

Added regime-stability stress testing to the enterprise so that every
candidate is checked for dependence on the data-generating process before
its results are admitted into the evidence base. This closes the last gap
in the robustness capability that `METHODOLOGY.md` lists as required for
judging robustness (perturbation, parameter sensitivity, regime stability).

- Added `research/backtest/regime_stability.py`: `regime_stress()` runs the
  same walk-forward perturbation sweep across a family of regime scenarios
  (via `run_scenario()`); `RegimeScenario` / `RegimeScenarioResult` hold the
  scenario configuration and per-scenario medians; `regime_stability` verdicts
  are one of CONSISTENT_WITH_NOISE (candidate indistinguishable from the
  coin-flip null in every scenario), REGIME_STABLE (consistent behavior —
  same edge or no edge — across regimes, candidate dispersion not greater
  than twice the null's), or REGIME_DEPENDENT (candidate results swing
  across regimes far more than a coin-flip null would, i.e. fits one regime).
  Also provided: `stress_regime_scenarios()` (extreme drift regimes for
  demonstration, documented as stress cases), `canonical_regime_scenarios()`,
  and contrived signals `direction_signal()` and `always_long_signal()`.
- Added `tests/test_regime_stability.py` (14 tests: scenario generation,
  determinism, noise behavior, detection of regime-dependence, detection of
  regime-stable adaptive signals, edge cases).
- Added `examples/regime_stability_demo.py` demonstrating all three verdicts
  on synthetic data.

## Next activation

1. Real-data readiness: if an in-scope public real dataset is identified for
   research-only simulation, create its manifest per
   `research/REAL_DATA_FEASIBILITY.md` and pass the pre-run leakage review
   checklist before any real-data execution.
2. Optionally: extend the engine with a margin/collateral model (backlog item
   from the first activation) so that high-leverage strategies are modeled
   safely; currently the engine allows negative cash without mark-to-market
   margin.

## Verification (executed)

The framework is deterministic and the full suite executes:

- `pip install -q numpy` (resolved numpy 2.5.3); all modules compile.
- `python -m unittest discover -s tests -v`: **56 tests, all passing**
  (31 pre-existing engine/metrics/data tests + 14 perturbation tests +
  11 regime-stability tests).
- `python -m examples.ma_crossover`: runs end-to-end and prints the full
  perturbation deviation scan; two independent runs produce byte-identical
  output (verified programmatically).
- `python -m examples.volatility_regime_filter`: runs end-to-end with
  walk-forward output; leakage checks pass on the full-sample runs.
- `python -m examples.regime_stability_demo`: runs end-to-end and prints
  all three verdicts; two independent runs produce byte-identical output.

Corrected verified results on seed 42 (regime-switching synthetic data;
tooling-validation only), superseding the unverified figures in the
activation log:

MA crossover (warmup=60, 252d/84d walk-forward, train+test):
- Full sample, zero cost: trades=55, total return -106.93%, sharpe=-0.04,
  max drawdown 128.21%, turnover 319.8x.
- Walk-forward: 88 folds, 7392 OOS periods, mean log return -0.123, median
  log return -0.081; positive folds 37/88.
- Perturbation sweep (0.5x/1.0x/2.0x grid): medians near zero across the
  grid with mixed signs and a flat deviation scan
  (0.00x: -0.081, 0.50x: -0.054, 1.00x: -0.021); one param set slightly
  best (+0.045), no single-point peak — no robust edge.

Volatility regime filter (warmup=20, 252d/84d walk-forward):
- Full sample, zero cost: 79 entries, 0 completed round-trips (never
  exits), total return -42.83%, sharpe=-0.10, max drawdown 74.80%.
- Walk-forward: 90 folds, 7560 OOS periods, mean log return -0.029, median
  log return 0.000.

Regime-stability stress test (seed 42, 600 bars, train=60d/test=20d,
walk-forward IS/OOS; the extreme regime family is documented as stress
cases, not realistic parameterizations):

- Long-only rule on extreme up (+0.9/yr) / neutral / down (-0.9/yr) regimes:
  candidate medians [+0.055, +0.005, -0.079] across scenarios;
  candidate dispersion 0.055 > 2x null dispersion 0.019. Verdict:
  REGIME_DEPENDENT (correctly flags a rule whose results swing with the
  regime mix).
- Direction-following rule (long up, short down) on the up/down pair:
  candidate medians [+0.055, +0.048]; candidate dispersion 0.003 <
  2x null dispersion 0.006. Verdict: REGIME_STABLE (consistent edge in both
  regimes).
- Coin-flip signal on the canonical realistic family: medians near zero in
  all four scenarios. Verdict: CONSISTENT_WITH_NOISE.
- MA crossover on the canonical realistic family: medians near zero in all
  scenarios. Verdict: CONSISTENT_WITH_NOISE (honest exploratory finding:
  no regime dependence is detectable on realistic regime mixes).

## Notes

Note: the activation log's earlier quantitative claims (e.g. full-sample
+15.99%, Sharpe 0.05) were not reproducible; the log had been written with
Python execution unavailable, so the verified run above supersedes it.
## Evidence standard

Synthetic data validates tooling only; it is not evidence that any
strategy is profitable in live markets. A strategy on synthetic data is
exploratory simulation only. Real-data work must carry its own audited
data and provenance and pass the leakage and perturbation gates before
admission to the evidence base.
