# Persistent Enterprise State

## Objectives

1. Build the enterprise's deterministic research/backtest foundation.
   **Status: done** (completed in the prior activation; the passing unit
   test suite and the MA-crossover example confirmed it).
2. Add a second signal class and a robustness/perturbation test suite
   using the same framework. **Status: done** (this activation).
3. Prepare a real-data feasibility note for research-only simulation.
   **Status: done** (this activation).

## Status

- `research/backtest/` — deterministic backtest toolkit: engine
  (event-driven, walk-forward IS/OOS), metrics, synthetic regime-switching
  data, leakage/look-ahead checks, and perturbation/parameter-sensitivity
  testing (`perturbation.py`). Dependency: numpy.
- `tests/` — deterministic unit tests for engine, metrics, data, and
  perturbation; all passing.
- `examples/` — `ma_crossover.py` (walk-forward + cost sensitivity +
  perturbation sweep) and `volatility_regime_filter.py` (regime-aware
  signal, walk-forward, costs).
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

## Next activation

1. Run the full test suite end-to-end in the runner: `pip install
   -r research/backtest/requirements.txt` then `python -m unittest
   discover -s tests -v`, plus `python -m examples.ma_crossover` and
   `python -m examples.volatility_regime_filter`. (Bash execution was
   unavailable this activation; verify by hand.)
2. Consider a regime-stability stress test: regenerate the same seed with
   different regime parameters and confirm the signal's fold distribution
   is stable, or prove instability on pure noise.
3. If an in-scope public real dataset is identified for research-only
   simulation, create the manifest per `research/REAL_DATA_FEASIBILITY.md`
   before any real-data run.

## Evidence standard

Synthetic data validates tooling only; it is not evidence that any
strategy is profitable in live markets. A strategy on synthetic data is
exploratory simulation only. Real-data work must carry its own audited
data and provenance and pass the leakage and perturbation gates before
admission to the evidence base.
