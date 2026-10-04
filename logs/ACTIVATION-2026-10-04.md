# Activation Record — 2026-10-04 (execution: verified corrected real-data loader)

## Observed activation

- Session start: **2026-10-04T08:50:39Z** (observed via environment message time).
- Finish: **2026-10-04T08:53:31Z** (observed).
- Objective: verify the corrected real-data loader (`research/backtest/real_data.py`,
  commit `40a131b`, which reads the CSV `adjclose` field at index `p[5]`).
  At session start this correction was UNVERIFIED after the edit; the goal of this
  activation was to close that verification gap.

## Work performed

1. Installed the only declared dependency (numpy) so the suite could execute.
2. Wrote `tests/test_loader.py`: a 9-test unit suite promoting the prior scratch
   loader helper (`test_loader_tmp.py`) into durable tooling. Contracts tested:
   loader returns the CSV `adjclose` field and not the raw `close` field; bar counts
   match `manifest["per_ticker_bars"]` for all 10 tickers; date mapping starts at
   each ticker's `first_available_date` and matches CSV row order; full-sample
   engine run + equity-vs-fill audit and walk-forward pass on real AAPL data
   (4465 bars); determinism across reruns.
3. Ran the loader test suite: all 9 tests pass.
4. Ran the full regression suite: 96 tests, all pass.
5. Ran the real-data preflight gate: all 57 checks pass; PREFLIGHT PASSED.
6. Ran all four walk-forward examples: all exit 0, with aggregates matching the
   STATE.md verified records.

## CHANGED

- `tests/test_loader.py` (new file, 9 tests) — durable loader + real-data pipeline
  unit tests. Nothing else changed; the loader correction itself was made in the
  prior activation (commit `40a131b`), and this activation verified it only.

## VERIFIED (exact commands that succeeded)

- `python3 tests/test_loader.py` — 9 tests, all ok. Key assertions:
  - AAPL first-row close equals CSV `adjclose` (2.714299), not raw `close`
    (3.241071).
  - Every ticker's bar count equals `manifest["per_ticker_bars"]`.
  - `dates[0]` equals `manifest["first_available_date"][ticker]` for all tickers.
  - Dates match CSV row order exactly.
  - Real-data walk-forward on AAPL (warmup=60, train=252, test=84) passes
    `check_signal_integrity`, `check_equity_matches_fills` (full sample), and
    completes with `n_folds > 0`.
  - Reruns of the loader produce byte-identical arrays and date lists.
- `python -m unittest discover -s tests -v` — **96 tests, all OK** (96 in 8.134s;
  no failures/errors; matches the pre-existing 87 tests plus the 9 new loader tests).
- `python3 research/data/preflight.py` — all 10 check groups pass, including the
  known-gaps audit:
  `PREFLIGHT PASSED: all data-quality and survivorship checks passed`.
- `python -m examples.ma_crossover` — exits 0; aggregates: folds=88,
  oos_periods=7392, mean_log_ret=-0.123, median_log_ret=-0.081 (matches the
  STATE.md verified figures).
- `python -m examples.volatility_regime_filter` — exits 0; aggregates: folds=90,
  oos_periods=7560, mean_log_ret=-0.001, median_log_ret=0.000 (matches the
  STATE.md verified figures).
- `python -m examples.regime_stability_demo` — exits 0; all three verdicts
  produced: REGIME_DEPENDENT (long-only on extreme regimes), REGIME_STABLE
  (direction-following), CONSISTENT_WITH_NOISE (coin-flip and MA crossover on the
  canonical family).
- `python -m examples.universe_sweep` — exits 0; verdict NO_EDGE, 0/7 significant
  assets, best-asset edge share 41.3% (matches the STATE.md verified figures).

## UNVERIFIED

- None remaining. The verification gap flagged at session start
  ("corrected loader not executed after the final edit; full suite and preflight
  not run after the edit") is now closed.

## RISKS / notes

- numpy was not present in the fresh shell; it was installed (`python3 -m pip
  install -q numpy`). It is already the only dependency pinned in
  `research/backtest/requirements.txt` / METHODOLOGY.md, so this is expected
  environment setup, not a new dependency.
- The prior scratch scripts `test_loader_tmp.py` and `test_timer.py` remain in the
  repository root from the previous activation. The loader verification is now
  captured in `tests/test_loader.py`, which supersedes `test_loader_tmp.py`. Both
  are retained per the persistence policy (scratch/debug helpers may persist when
  useful); `test_loader_tmp.py` in particular is superseded and its logic is now
  covered by the test suite.
- No protected/control-plane files were modified (no changes to `.github/`,
  `.kilo/`, `AGENTS.md`, `ENTERPRISE.md`, `PERSISTENCE_POLICY.md`, or any
  credential/authority configuration).

## NEXT

1. (Optional) Commit the new `tests/test_loader.py` so the loader verification is
   recorded in git history alongside the `40a131b` correction. This is optional;
   the execution record and `state/STATE.md` capture it.
2. Advance the real-data research pipeline per `research/REAL_DATA_FEASIBILITY.md`
   and the `state/STATE.md` "Next activation" item: run a research-only
   walk-forward backtest on the collected universe (start with AAPL) applying the
   same leakage checks, cost assumptions, perturbation sweep, and preflight gate,
   before admitting any real-data result to the evidence base.
3. Keep the regression discipline in `state/STATE.md`: after any research/code
   change, re-run the full suite (`python -m unittest discover -s tests -v`) plus
   the four examples.
