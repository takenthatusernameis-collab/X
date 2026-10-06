# Activation Log — 2026-10-06

## Activation — 37414038601 (breakout continuation frontier cell)

**Environment observation:** message time 2026-10-06T04:45:51Z (runner wall clock); run identity GITHUB_RUN_ID=37414038601, GITHUB_RUN_ATTEMPT=1, SHA=a29ba2245a0e97007a63f3814bae9b5263204d63, REF_NAME=main, repository root=/home/runner/work/X/X. Branch checkout: clean `main`, up to date with origin/main.

**Objective (task R-004):** Does a fixed 20-day breakout continuation signal (long when the close exceeds the highest close of the preceding 20 trading days; hold 1 day; daily rebalance) produce reproducible evidence beyond the currently qualified momentum family?

**Selection basis:** frontier-first. This is the genuinely-new-signal-class cell the frontier's Next action called for ("a fresh activation should propose a genuinely new signal class ... and carry a fresh a-priori falsification prediction").

**A-priori falsification prediction:** the breakout filter is a directionally-biased selection of the momentum long leg; expected CONSISTENT_WITH_NOISE in most assets, with possible small REGIME_STABLE positive edges only where the momentum edge already exists (AAPL, MSFT, GOOGL, AMZN, META, TSLA).

## Work performed (chronological)

1. **Read / preflight:** LEARNING_STATE.md, STATE.md, campaign task_queue.json, activation_status.json, worker_progress.md, the manual-deep-grind engineering record; confirmed clean `main` checkout and working `python3` / git. 156/156 regression tests pass before changes.

2. **Framework addition:** added `research/backtest/regime_stability.py::breakout_signals()` (past-only breakout continuation, same contract as `momentum_signals`) and exported it via `research/backtest/__init__.py`.

3. **Smoke test (representative path):** `research/checks/smoke_breakout_signals.py` on AAPL — signals generated for lookbacks 5/10/20, leakage PASS, one-regime-segment stress_segments run exits 0 (CONSISTENT_WITH_NOISE, 28 folds).

4. **Main check:** `research/checks/breakout_20day_cont.py` — manifest 10/10 OK, preflight PASSED, leakage PASS on AMZN/JPM (signals integrity + equity-fill audit), regime-stability gate across all 10 tickers with a matched coin-flip null, lookbacks {5, 10, 20}, bounded sensitivity, determinism r1==r2, artifact written to `state/check_artifacts/breakout_20day_cont_results.json`.

5. **Independent verification:** `research/checks/verify_breakout_20day_cont.py` — GATE 1 (AMZN/JPM fresh walk_forward + noise_benchmark on lookbacks 5/10/20) MATCH; GATE 2 (all-asset fresh medians/nulls + regenerated verdicts from unrounded values) MATCH; GATE 3 (determinism) identical.

6. **Unit tests:** 5 new tests added to `tests/test_regime_stability.py` (contract, past-only, determinism, short-series edge case, regime-stress integration); regression suite 161/161 passing, no regressions.

7. **Records:** verdict FALSIFIED written to `state/STATE.md`, `state/LEARNING_STATE.md` (frontier table + learning history), `state/activation_status.json`, `state/worker_progress.md`; this activation log.

## Results

- Lookback regime-stability verdict counts (10-asset universe): lookback 5 — REGIME_STABLE=3, CONSISTENT_WITH_NOISE=7, REGIME_DEPENDENT=0; lookback 10 — REGIME_STABLE=1, CONSISTENT_WITH_NOISE=8, REGIME_DEPENDENT=1; lookback 20 (canonical) — REGIME_STABLE=1, CONSISTENT_WITH_NOISE=9, REGIME_DEPENDENT=0.
- Sensitivity verdicts: ROBUST=1 (TSLA, edges at lookbacks 5 and 20), SENSITIVE=2 (AAPL @5, NVDA @10), NO_EDGE=7 (MSFT, GOOGL, AMZN, META, JPM, JNJ, XOM).
- Borderline positive verdicts: AAPL @5 [0.039, 0.021, 0.063, 0.021], TSLA @5 [0.038, 0.013, 0.068] + @20 [0.019, 0.040, 0.053], NVDA @10 [0.003, 0.047, 0.050] — small magnitude, non-lookback-robust (except TSLA), and absent from the momentum-asset list.

## Verdict

**FALSIFIED** — the breakout continuation rule produces no reproducible robust edge on the collected universe. The result matches the a-priori prediction (noise-like in most assets; small positive verdicts only among momentum assets); the breakout filter adds no robust evidence beyond the momentum family. The momentum frontier (admitted as candidate positive evidence) is unchanged; the frontier-first delta remains RETAIN (demonstrated a sixth time: right cell selected, decisive verdict, independent verification MATCH, no equivalent re-tests).

## Change summary (what actually changed)

- `research/backtest/regime_stability.py` — new `breakout_signals()`.
- `research/backtest/__init__.py` — export of `breakout_signals`.
- `research/checks/breakout_20day_cont.py` — new check.
- `research/checks/verify_breakout_20day_cont.py` — new independent verifier.
- `research/checks/smoke_breakout_signals.py` — scratch smoke test (retained; useful diagnostic).
- `tests/test_regime_stability.py` — 5 new unit tests.
- `state/check_artifacts/breakout_20day_cont_results.json` — new artifact.
- `state/STATE.md`, `state/LEARNING_STATE.md`, `state/activation_status.json`, `state/worker_progress.md`, `logs/ACTIVATION-2026-10-06.md` — records.

## Verification (exact post-change checks)

- `python3 research/checks/smoke_breakout_signals.py` — exits 0.
- `python3 research/checks/breakout_20day_cont.py` — exits 0; manifest 10/10; preflight PASSED; leakage PASS; artifact written; determinism r1==r2.
- `python3 research/checks/verify_breakout_20day_cont.py` — GATE 1/2/3 all PASS (AMZN/JPM MATCH on lookbacks 5/10/20; all-asset fresh medians/nulls + regenerated verdicts MATCH; determinism identical).
- `python3 -m unittest discover -s tests -v` — 161 tests, all passing (156 existing + 5 new), 0 failures.

## Unverified

- No per-checkpoint UTC timestamps captured (the `date` shell form is denied in this environment); run identity anchored to GITHUB_RUN_ID=37414038601.
- Threshold sensitivity of the three borderline REGIME_STABLE verdicts (medians just above the framework's 0.05 band) is not independently quantified; flagged as a caveat.
- No KILO_RECOVERY_BRANCH present; no outstanding recovery branch inspected.

## Next

Task R-001: cost-sensitivity test of the qualified momentum edge (fixed cost model, lookback 3/5/10, matched null) through the full gate stack; report whether conservative costs erase the edge.
