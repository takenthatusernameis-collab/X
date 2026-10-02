# Research

This directory holds non-secret quantitative research methods,
deterministic experiment definitions, result summaries, validation
artifacts, and supporting documentation.

## Structure

- `backtest/` — deterministic backtest toolkit (engine, metrics,
  synthetic data, leakage checks). Requirements: `numpy`.
- `examples/` — reproducible demonstrations of the framework.
- `METHODOLOGY.md` — methodology, evidence standard, and discipline rules.

## Running experiments

1. Ensure the toolchain is installed:

   ```bash
   pip install -r research/backtest/requirements.txt
   ```

2. Run a deterministic example:

   ```bash
   python -m examples.ma_crossover
   ```

3. Run the test suite:

   ```bash
   python -m unittest discover -s tests -v
   ```

## Evidence standard

Synthetic data validates the tooling only. Real-data claims require
audited data, walk-forward IS/OOS validation, cost modeling, and
leakage checks. See `METHODOLOGY.md`.
