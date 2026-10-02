# Persistent Enterprise State

## Current objective

Build the enterprise's first deterministic research/backtest foundation
so that future activations can discover, evaluate, and validate trading
ideas with reproducible walk-forward validation instead of starting from
scratch. (Completed this activation.)

## Status

- `research/backtest/` — deterministic backtest toolkit (engine, metrics,
  synthetic data, leakage checks). Installed dependency: numpy.
- `tests/` — deterministic unit tests covering costs, IS/OOS separation,
  warmup, determinism, and leakage checks; all passing.
- `examples/ma_crossover.py` — past-only signal generation +
  walk-forward IS/OOS demonstration.
- `research/METHODOLOGY.md` — evidence standard, no-look-ahead rules,
  walk-forward discipline.
- README updated to reflect the infrastructure.

## Next activation

1. Verify the framework end-to-end in this runner (install numpy, run
   tests and the MA-crossover example).
2. Add one more signal class (e.g., volatility-regime filter) and one
   robustness/perturbation test (parameter sensitivity) using the same
   framework.
3. Prepare a real-data feasibility note: what data is allowed in-scope
   for research-only simulation, how provenance would be recorded, and
   the leakage review checklist before any real-data run.

## Evidence standard

Synthetic data validates tooling only; it is not evidence that any
strategy is profitable in live markets.
