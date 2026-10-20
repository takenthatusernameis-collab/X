# Research Methodology

Research/simulation only. This repository is not a live-trading or
production-execution system. No exchange credentials, production
secrets, or personal files belong here.

## What is built

- `research/backtest/`: deterministic backtest toolkit.
  - `data.py`: synthetic daily bar generator (tooling validation only).
  - `engine.py`: event-driven daily backtest with IS/OOS walk-forward.
  - `metrics.py`: deterministic performance/risk metrics.
  - `leakage.py`: look-ahead and fill-equity integrity checks.
- `tests/`: deterministic unit tests for engine, metrics, and data.
- `examples/`: reproducible demonstrations of the framework.

## Equivalence principle

Everything here is synthetic and is used to validate the tooling
(engine, metrics, leakage checks). Synthetic data cannot establish the
robustness of a live strategy. Any future real-data work must bring its
own audited data and provenance; the same tooling is used.

## No-look-ahead rules (enforced by the engine)

1. **Close-only pricing.** Every fill executes at `close +
   fixed_slippage + proportional_slippage`. Open/high/low are never
   used for execution, so no intrabar lookahead is possible.
2. **Warmup discipline.** The first `warmup_periods` bars hold initial
   capital unchanged and no signal is used. Signal authors must pad
   their signal series (e.g., neutral before the slow moving average)
   so no real signal exists before the warmup boundary.
3. **Signal dates.** Every signal must reference a bar date that
   exists in the series; the engine and `check_signal_integrity()`
   reject signals pointing to future bars.
4. **Fill audit.** `check_equity_matches_fills()` recomputes equity
   from the fill sequence and asserts it matches the engine's equity
   curve, so no equity change is untraceable to a recorded fill.

## IS/OOS walk-forward discipline

- The series is split into folds: `[warmup, warmup + train,
  warmup + train + test]`, sliding across the series.
- Only the **test** segment is backtested and reported; the train
  segment represents information available at fold start.
- Costs and cash policy are identical across folds; the seed and
  window sizes are fixed so results are reproducible.
- Aggregates report the distribution of OOS fold returns, not a single
  full-sample number. Full-sample results are reported separately as
  a reference only.

## Reproducibility

- All generators take a `seed`; every run prints the parameters used
  (see examples).
- No randomness is drawn after a seed is set; results are deterministic.
- `requirements.txt` pins the only dependency (numpy); record the
  resolved version in any validated run summary.

## Evidence standard

- Tooling validation with synthetic data = **validated tooling**.
- A strategy on synthetic data = **exploratory simulation only**.
- A strategy on real, audited data with walk-forward costs and leakage
  checks = candidate evidence, still to be judged for robustness
  (perturbation, parameter sensitivity, regime stability).
