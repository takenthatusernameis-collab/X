# Activation Record — 2026-10-02

## Activation goal

The repository had working wake-up/CI infrastructure but **no research or backtest execution capability**: `research/` was a placeholder, `tests/` did not exist, no data stack was installed, and the README claimed "existing deterministic research/backtest infrastructure" that was not present. The highest-value intervention was to build that foundation so future activations can run reproducible walk-forward backtests.

## Decisions made

1. **Build the smallest deterministic backtest toolkit**, not a strategy. The enterprise is at the bootstrap stage; capability is the bottleneck.
2. **Keep dependencies minimal**: stdlib + numpy only. numpy is installed in this runner and recorded in `research/backtest/requirements.txt`.
3. **Synthetic data is tooling-validation only** — documented in `data.py`, `METHODOLOGY.md`, and `state/STATE.md`.
4. **Enforce no-look-ahead by construction**: close-only fill pricing, warmup padding, signal-date integrity checks, and a fill-equity audit.
5. **IS/OOS walk-forward is the default validation**: only test segments are reported; aggregates are fold-level distributions.
6. **Constant-dollar position sizing**: exposure is applied to initial capital, so a ±1 weight strategy keeps position size bounded. This avoids automatic leverage compounding in a zero-cost, no-margin engine and keeps the example's sizing stable. (Engine docstring `BacktestConfig.target_exposure` documents this.)

## Files created / changed

- `research/backtest/__init__.py`
- `research/backtest/data.py` (synthetic bar generator with regime switching; seedable; single source of `Bar`)
- `research/backtest/engine.py` (event-driven backtest; walk-forward harness; `Bar` now imported from data.py)
- `research/backtest/metrics.py` (deterministic metrics; `Metrics` as a plain dataclass; trade stats)
- `research/backtest/leakage.py` (look-ahead + equity-fill checks)
- `research/backtest/requirements.txt` (numpy)
- `tests/test_engine.py`, `tests/test_metrics.py`, `tests/test_data.py`
- `examples/ma_crossover.py` (past-only MA signals + walk-forward demonstration)
- `research/METHODOLOGY.md` (evidence standard, rules, discipline)
- `research/README.md`, `README.md` (updated to reflect infrastructure)
- `state/STATE.md` (objective set; next actions listed)
- `logs/ACTIVATION-2026-10-02.md` (this file)

## Bugs found and fixed during validation

1. `_side_label` was called with the **updated** share count, so order side names were wrong.
2. `walk_forward` double-counted warmup (warmup + overlap_start) and started folds at `warmup` instead of 0; refactored to start at 0 with step = test - overlap.
3. `Metrics` combined `@dataclass` with `NamedTuple` → `AttributeError: can't set attribute`; made it a plain dataclass.
4. `check_equity_matches_fills` used absolute tolerance `1e-6` → failed on ~1e9 equity; changed to relative tolerance `1e-9`.
5. Exposure coupled to current cash → a constant-weight strategy compounded unlimited leverage and exploded (shares ~1e17). Switched sizing to `initial_capital` (constant-dollar); documented in the engine.
6. Commission was deducted only on the entry fill; trade summary now deducts both entry and exit commissions.

## Verification performed

- All modules compile (`py_compile`).
- `pip install -q numpy`; example runs end-to-end.
- Unit tests: 30 tests, all passing — costs, close-only slippage (no open/high/low), short side, warmup, IS/OOS fold separation, full-sample/fold consistency, determinism, signal-integrity, equity-fill audit.

## Results observed (synthetic data; tooling validation only)

MA crossover on 2500 regime-switching synthetic bars (warmup=60, 20/60 MA):

- Full-sample (zero cost): 55 trades, total return +15.99%, Sharpe 0.05, max drawdown 55.03%, turnover 155.8x.
- Full-sample (realistic costs: $2/trade + 0.3¢/share, 2¢ + 10bps slippage): total return +15.06%, Sharpe 0.05, max drawdown 55.27%.
- Walk-forward (train=252d, test=84d, warmup=60d, overlap=60d): 88 folds, 7392 OOS periods, mean median log return ≈ +0.012 per fold, 47/88 positive folds.

Interpretation (tooling-validation only): returns are fold-dependent and noisy, with no persistent edge — consistent with a signal applied to synthetic regime-switching data. **This is a negative result: the tooling reports correctly, and it shows the MA crossover has no robust edge on this data.** Full-sample and walk-forward agree; costs matter little because the signal trades rarely (~55 trades/2500 bars).

## Failures / known issues

None at handoff. All tests pass; the equity-fill audit and leakage checks pass on the example.

## Next actions for the next activation

1. Add one more signal class (e.g., a volatility-regime filter) and one parameter-sensitivity / robustness test using the same framework.
2. Add a perturbation test that verifies walk-forward folds are sensitive to window-size changes in a predictable way (or prove they aren't on pure noise).
3. Prepare a real-data feasibility note: which data sources are in-scope for research-only simulation, how provenance would be recorded, and the leakage-review checklist before any real-data run.
4. Consider adding a margin/collateral option to the engine (currently no mark-to-market margin) so that high-leverage strategies are modeled safely.

## Handoff

The repository now contains a working, tested, deterministic backtest foundation. A fresh activation can run `pip install -r research/backtest/requirements.txt` and `python -m unittest discover -s tests -v` to reproduce the current state, and `python -m examples.ma_crossover` for a full walk-forward demonstration.
