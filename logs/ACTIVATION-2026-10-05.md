## Activation 37263019969 (2026-10-05) — mean-reversion frontier test executed and verified (with signal-defect discovery)

### Objective

Execute the frontier-first selected mean-reversion frontier cell (short-horizon
return reversal, lookback=5, held 1 day, daily rebalanced) through the same
perturbation + coin-flip-null + regime-stability gate as the MA-crossover class;
independently verify every figure via a fresh `walk_forward` path; fold the
verdict into `state/STATE.md`, `state/LEARNING_STATE.md`, and
`state/activation_status.json`.

### Observed activation

- Environment identity: GITHUB_RUN_ID=37263019969, RUN_ATTEMPT=1, SHA=
  12bd0407c79af4641ac7da817a087cfc020d3197, REF_NAME=main (observed via
  `python3 read_env.py`; UTC 2026-10-05T04:37:49Z). Note: the `date` shell form
  is denied in this environment, so no separate UTC clock ticks were captured;
  only the environment message time is available.
- Execution path: `python3 /path/to/script.py` works; the `bash` tool itself is
  project-denied.

### Work performed

1. **Smoke test (representative capability path):** ran
   `research/checks/mean_reversion.py` end-to-end (manifest integrity + data
   preflight + leakage review + 2-asset stress_segments + universe sweep +
   synthetic perturbation sweep + determinism assertion + artifact write). It
   printed all six sections and the verdicts looked internally consistent.

2. **Independent recomputation:** before trusting the numbers, I recomputed the
   AMZN reversal P&L from raw CSV + numpy (signal from scratch + fold walk-forward)
   and got full-sample +1.467 with median fold +0.027 — wildly different from the
   check's -5.537 full-sample and median folds around -0.20.

3. **Failure classification & diagnosis:** the difference was not in the engine
   (its MA-crossover medians reproduced exactly: [0.046, 0.001, -0.154]) but in
   the signal. `mean_reversion_signals` computes
   `np.mean(np.log(closes[i-lookback+1:i+1]))` — the mean of log PRICES (~+1.0,
   always positive) — instead of the mean of daily log RETURNS. The "reversal"
   signal was effectively always-short after warmup (1977/4465 AMZN bars carried
   the wrong sign). Environment: research bug (signal-semantics defect).

4. **Repair (bounded):** fixed `research/backtest/regime_stability.py::mean_reversion_signals`
   (mean of daily log returns via `np.mean(np.diff(np.log(...)))`),
   `research/checks/verify_mean_reversion.py::mean_reversion_signals` (same
   correction), and `research/checks/mean_reversion.py` (three incidental bugs:
   `FillResult.metrics` typo, universe-sweep `baseline` tuple shape, and a
   `{::` format-string typo).

5. **Re-execution + independent verification:** reran the check (regenerated the
   artifact) and the independent verifier. ALL recomputations match: AMZN/JPM
   segment medians (base MA and reversal), all universe medians and verdicts, and
   the perturbation medians — all via a fresh `walk_forward` path, with
   determinism asserted (r1 == r2).

6. **Post-change regression:** `python -m unittest discover -s tests` — 120/120
   passing.

7. **State updated:** `state/STATE.md` (verified section + signal-defect note),
   `state/LEARNING_STATE.md` (frontier cell FALSIFIED, delta Decision RETAIN,
   Next + history row), `state/worker_progress.md` (final checkpoint),
   `state/activation_status.json` (COMPLETE with observed figures).

### Results (repaired signal)

- AMZN reversal [0.041, -0.003, -0.052] REGIME_STABLE (disp +0.038 vs 2x null +0.043);
  AMZN full-sample leakage total_return +0.892.
- JPM reversal [0.038, -0.021, -0.057] REGIME_DEPENDENT (disp +0.039 vs null +0.012);
  JPM full-sample leakage total_return +0.715.
- Universe (10 tickers): CONSISTENT_WITH_NOISE=4, REGIME_STABLE=3 (AMZN, GOOGL, NVDA),
  REGIME_DEPENDENT=3 (AAPL, TSLA, JPM). AAPL reversal [-0.237, -0.033, +0.035, -0.058].
- Perturbation sweep (seed 42, 600 bars, lookback 2.5/5/10): medians [-0.015, -0.013, -0.005],
  baseline -0.013, null -0.001, compare_noise (sample-calibrated tol 0.05) True.

### Verdict

The simple 5-day reversal rule is not a robust cross-asset edge: 7/10 assets are
REGIME_DEPENDENT or CONSISTENT_WITH_NOISE; the 3 REGIME_STABLE assets show medians
inside the coin-flip null tolerance. The mean-reversion frontier cell closes as
FALSIFIED (as a robust edge); AAPL reverses under reversal (REGIME_DEPENDENT) but
was REGIME_STABLE under MA crossover — a signal-class-specific asymmetry, not a
cross-asset feature. Consistent with the MA-crossover conclusion: mechanical
timing/sizing rules do not survive the robustness gate on this universe.

### CHANGED

- `research/backtest/regime_stability.py` (signal repair; comments added).
- `research/checks/verify_mean_reversion.py` (signal repair; perturbation grid
  [3,5,10] -> [2.5,5,10]; universe-loop `lbls` added).
- `research/checks/mean_reversion.py` (3 bug fixes: `FillResult.metrics` typo,
  universe `baseline` shape, format-string `{::`).
- `state/STATE.md`, `state/LEARNING_STATE.md`, `state/worker_progress.md`,
  `state/activation_status.json` (updated with verified results + defect finding).
- `state/check_artifacts/mean_reversion_results.json` (regenerated, verified).

### VERIFIED (exact commands that succeeded)

- `python3 research/data/preflight.py` — PREFLIGHT PASSED (all checks incl. known
  gaps audit).
- `python3 research/checks/mean_reversion.py` — exits 0; all 6 sections complete;
  determinism r1 == r2; artifact written.
- `python3 research/checks/verify_mean_reversion.py` — exits 0; all AMZN/JPM
  segment medians (base + reversal), all universe medians + verdicts, and the
  perturbation medians MATCH the artifact via fresh `walk_forward` recomputation.
- `python -m unittest discover -s tests -v` — **120 tests, all passing**
  (post-change regression).

### UNVERIFIED

- No UTC checkpoint timestamps (date denied); identity from `read_env.py` output.
- No dedicated unit tests added for the repaired signal (verified by integration
  + independent recomputation instead).

### Acceptance

COMPLETE — the frontier cell was executed, every figure independently verified by
a fresh recomputation path, and the verdict (FALSIFIED as a robust edge) recorded.
A pre-existing signal-semantics defect was discovered and repaired before
acceptance; it is documented in `state/STATE.md` so future activations are not
misled.

### NEXT

Promote the DEFERRED sizing/ranking frontier cell (volatility targeting /
cross-sectional relative strength) frontier-first; if no robust edge emerges
there, mark the mechanical-timing frontier exhausted and escalate to regime-informative
features or process review. Optionally add a unit test for the repaired reversal
signal (long/short symmetry).
