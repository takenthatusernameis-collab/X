# Activation Record — 2026-10-05

## Activation goal

Extend the fold-level statistical analysis from the single-asset diagnostic
(`research/checks/verify_aapl_stats.py`) into a reusable **per-asset
t-statistic over the collected universe**, the optional follow-on #3 listed
in `state/STATE.md`. The result should be a durable, deterministic helper
that reports per-asset mean/median/std of OOS fold log returns, t-statistic
against H0: mean log fold return = 0, degrees of freedom, 95% Wald CI, and a
cross-asset summary counting assets significant at nominal 5% and those
surviving a Bonferroni family-wise correction — giving a multiple-testing-aware
significance measure complementary to the coin-flip null in `universe.py`.

## Observed activation

- Session start: **2026-10-05T00:13:54Z** (observed via environment message
  time).
- Finish: approximately **2026-10-05T00:17 UTC** (estimate; the `date` shell
  form is denied in this environment, so only the start timestamp is exact).

## Work performed

### 1. Baseline re-verified before change

- `python -m unittest discover -s tests -v`: **120 tests, all passing** —
  the claimed baseline in `STATE.md` was independently confirmed green before
  writing new code.

### 2. Created `research/checks/universe_stats.py`

Deterministic checker (seed 42), patterned on `verify_aapl_stats.py` and
`determinism_check.py`:

- Loads the manifest (checksums) and runs the data preflight gate before
  any computation.
- Walk-forwards the MA(20/60) crossover on all 10 collected tickers
  (train=252d, test=84d, warmup=60d, overlap=60d) using the collected
  adjusted-close series.
- Per-asset output: n_folds, mean/median/std log fold return, positive-fold
  count, t-statistic vs H0: mean log fold return = 0, df, 95% Wald CI, and
  two significance flags (nominal 5%, and Bonferroni family-wise at
  alpha 0.05/10 = 0.005).
- Cross-asset summary: nominal 5% count and Bonferroni-surviving count.
- Built-in determinism re-run: `key_fields()` runs twice and asserts
  `r1 == r2`, mirroring the existing `determinism_check.py` pattern.
- Docstring documents the complementary role of the t-test vs the coin-flip
  null and states the synthetic/real-data evidence standard.

### 3. Executed and results

Manifest integrity: 10/10 OK. Preflight: PASSED (all 57 checks incl. known-
gaps audit). Walk-forward completed on all 10 tickers (n_folds 135–170,
fewer for META/TSLA because those series start later).

Per-asset fold statistics (seed 42):

| Ticker | n_folds | mean | median | std  | pos_folds | t_stat | 95% CI | sig_5% |
|---|---|---|---|---|---|---|---|---|
| AAPL | 170 | +0.0309 | +0.0242 | +0.1463 | 101/170 | +2.75 | [+0.0089, +0.0529] | yes |
| MSFT | 170 | -0.0200 | +0.0142 | +0.1768 | 93/170 | -1.48 | [-0.0466, +0.0065] | no |
| GOOGL | 170 | -0.0042 | +0.0091 | +0.1535 | 88/170 | -0.36 | [-0.0273, +0.0188] | no |
| AMZN | 170 | -0.0378 | +0.0188 | +0.2716 | 91/170 | -1.82 | [-0.0787, +0.0030] | no |
| META | 135 | +0.0309 | +0.0515 | +0.1361 | 82/135 | +2.64 | [+0.0079, +0.0539] | yes |
| NVDA | 170 | -0.3002 | +0.0291 | +3.0024 | 97/170 | -1.30 | [-0.7515, +0.1512] | no |
| TSLA | 154 | -1.1604 | -0.0452 | +5.3770 | 71/154 | -2.68 | [-2.0097, -0.3112] | yes |
| JPM | 170 | +0.0073 | +0.0077 | +0.1488 | 91/170 | +0.64 | [-0.0150, +0.0297] | no |
| JNJ | 170 | -0.0167 | -0.0150 | +0.1069 | 70/170 | -2.04 | [-0.0328, -0.0006] | yes |
| XOM | 170 | -0.0126 | -0.0207 | +0.1451 | 70/170 | -1.14 | [-0.0345, +0.0092] | no |

Cross-asset summary: **4 / 10 nominally significant at 5%** (AAPL +, META +,
TSLA -, JNJ -); **0 / 10 survive the Bonferroni family-wise correction**
(alpha = 0.005).

Determinism: `r1 == r2: True`; all per-asset fields identical across
independent re-runs.

### 4. Key observations (honest, exploratory)

- **AAPL cross-checks exactly** against the earlier independent diagnostic
  (`research/checks/verify_aapl_stats.py`): 170 folds, mean +0.0309, median
  +0.0242, std +0.1463, 101/170 positive, t=2.75 (df=169), 95% CI
  [0.0089, 0.0529]. The two independent implementations agree.
- **TSLA's significance is tail-driven, not a stable edge**: mean log return
  -1.1604 vs median -0.0452, std 5.377, CI [-2.01, -0.31]. An extreme
  negative tail dominates the mean and the CI is correspondingly wide. This
  is recorded as a negative/quality finding, not an edge.
- **The t-test and the coin-flip null diverge by design** for this candidate
  class: AAPL is t-significant (CI excludes 0) yet within the framework's
  sample-calibrated coin-flip noise band. That is precisely the complementary
  information both checks were built to report.
- **Nothing survives a family-wise correction** across the universe. No MA-
  crossover edge is a robust cross-asset feature under the multiple-testing-
  aware criterion.

### 5. State updated

`state/STATE.md` gained a "Current activation (per-asset t-statistic over the
collected universe)" section documenting the new tool, the verified per-asset
table, the cross-asset summary, and the negative assessment; the "Next
activation" list retains item 4 (REGIME_DEPENDENT deep-dive on AMZN, JPM).

## CHANGED

- `research/checks/universe_stats.py` (new file — per-asset walk-forward
  t-statistic checker with built-in determinism assertion).
- `state/STATE.md` — new "Current activation" section and research
  observations.

## VERIFIED (exact commands that succeeded)

- `python -m unittest discover -s tests -v`: **120 tests, all passing**
  (post-change regression check; same suite that existed before this edit).
- `python3 research/checks/universe_stats.py`: exits 0; all 5 sections
  complete; 10/10 manifest checksums OK; PREFLIGHT PASSED; per-asset table
  printed; `r1 == r2: True` on the determinism re-run.
- Cross-check: the AAPL row from `universe_stats.py` matches the earlier
  `verify_aapl_stats.py` figures (folds=170, mean +0.0309, median +0.0242,
  std +0.1463, t=2.75, df=169, CI [0.0089, 0.0529]).

## UNVERIFIED

- None material. Precise per-checkpoint and finish UTC timestamps could not
  be captured because the `date` shell form is denied in this environment;
  the start (2026-10-05T00:13:54Z) is exact, and the finish (~00:17 UTC) is
  an estimate.

## RISKS / notes

- Nominal per-asset significance counts are not corrected unless the
  Bonferroni column is used; the cross-asset summary always prints both.
- Fewer folds for META (135) and TSLA (154) reflect their later series
  start dates (2012-05-18 and 2010-06-29) and are handled naturally by walk-
  forward.
- TSLA's extreme tail (std 5.377) makes its t-statistic mean-driven; the
  median-based robustness tools in `regime_stability.py` / `universe.py` are
  the appropriate secondary lens for such assets.
- No protected/control-plane files were modified; no credentials or secrets
  accessed or persisted.

## NEXT

1. (Recommended) Regression discipline maintained: after any research/code
   change, re-run the full suite plus the examples; for this change:
   `python -m unittest discover -s tests -v` (120 OK) and
   `python3 research/checks/universe_stats.py` (determinism asserted internally).
2. Optional follow-on #4 from STATE.md: deep-dive on the REGIME_DEPENDENT
   assets (AMZN, JPM) — block-by-block breakdown of which regime mix drives
   the swing and a regime-filtered variant — the natural complement to this
   fold-level significance view.
3. Optionally fold the per-asset fold-return summary into
    `RegimeUniverseSummary` output so regime-stability runs report fold
    statistics alongside verdicts (item 3b from the prior activation).

## Activation — 02:21 UTC (regime-dependence deep-dive of AMZN and JPM)

### Objective

Deep-dive the two REGIME_DEPENDENT assets from the universe-level regime-
stability run (AMZN, JPM): characterize which regime mix drives the swing
(block-by-block breakdown) and test a regime-filtered variant (MA crossover
active only in turbulent segments). This is next-activation item 4 from
`state/STATE.md`.

### Observed activation

- Environment identity: GITHUB_RUN_ID=37255021664, RUN_ATTEMPT=1, SHA=
  53ecbf65de1a9b35a25a9665b42ef71d95562cf8, REF_NAME=main.
- Session start observed at 2026-10-05T02:21:14Z (from environment message
  time). Finish time cannot be captured precisely because the `date` shell
  form is denied in this environment.

### Work performed

1. Smoke test (representative path): `python3 examples/regime_stability_universe.py`
   executed end-to-end (manifest checksums, preflight, leakage review, regime
   classification, universe-level `stress_segments_across_tickers`). Results
   reproduce the documented universe figures exactly: AMZN
   medians [+0.046, +0.001, -0.154] REGIME_DEPENDENT, JPM
   [+0.109, -0.006, +0.033] REGIME_DEPENDENT.

2. Created `research/checks/regime_dependent_deep_dive.py`: block-by-block
   breakdown (segment dates/length/vol, candidate vs null median, folds,
   fold mean) plus the regime-filtered variant through the same
   `stress_segments` pipeline. Includes an internal independent recomputation
   of each segment median via `run_bars`, an assertion that segment count from
   the label scan matches the scenario count, internal determinism assertion,
   manifest integrity + preflight + leakage review.

3. Executed the deep-dive:
   - AMZN: [turbulent 2009-03-31..2013-06-10 (pos edge +0.046, 19/28 folds),
     calm 2013-06-11..2022-04-20 (no edge +0.001), turbulent 2022-04-21..2026-
     10-01 (neg edge -0.154)] -> REGIME_DEPENDENT (dispersion +0.086 vs 2x
     null +0.043).
   - JPM: same three segments [pos edge +0.109 (21/28), no edge -0.006, no
     edge +0.033] -> REGIME_DEPENDENT (dispersion +0.047 vs 2x null +0.024).
   - Regime-filtered variant (signals only in turbulent segments):
     AMZN still REGIME_DEPENDENT (dispersion +0.0859 vs 2x null +0.0854) —
     the swing sits entirely within the turbulent regime (+0.046 in 2009-2013,
     -0.154 in 2022-2026); JPM still REGIME_DEPENDENT (dispersion +0.0455
     vs 2x null +0.0051, ~9x the null).

4. Independent verification: `research/checks/verify_dd_independent.py`
   (fresh implementation via `walk_forward`, not `stress_segments`) recomputed
   all four runs (AMZN/JPM base + filtered). All recomputed medians MATCH the
   deep-dive figures exactly (to 3 decimals) and all four runs are deterministic
   across independent reruns.

5. Negative conclusion recorded: the regime filter does not rescue the MA
   crossover; the apparent edge is concentrated in one historical regime mix
   (2009-2013 post-crisis recovery) and reverses/disappears later. Both assets
   remain REGIME_DEPENDENT even for the filtered variant. This is regime-
   specific fitting, not a persistent regime-contingent edge. The MA crossover
   is reinforced as exploratory, not admitted to the evidence base.

### CHANGED

- `research/checks/regime_dependent_deep_dive.py` (new — block-by-block
  breakdown of AMZN/JPM plus regime-filtered variant).
- `research/checks/verify_dd_independent.py` (new — independent
  recomputation of the deep-dive figures via `walk_forward`).
- `state/STATE.md` — new "Current activation (regime-dependence deep-dive of
  AMZN and JPM)" section; next-activation item 4 marked done.

### VERIFIED (exact commands that succeeded)

- `python3 examples/regime_stability_universe.py`: exits 0; AMZN
  [+0.046, +0.001, -0.154] REGIME_DEPENDENT; JPM
  [+0.109, -0.006, +0.033] REGIME_DEPENDENT (reproduces documented figures).
- `python3 research/checks/regime_dependent_deep_dive.py`: exits 0; manifest
  10/10 OK; PREFLIGHT PASSED (57 checks); leakage PASS (AMZN, JPM); per-
  segment medians printed; internal determinism assertion passed.
- `python3 research/checks/verify_dd_independent.py`: exits 0; all four
  recomputed segment medians MATCH published deep-dive medians; all four runs
  deterministic.
- `python -m unittest discover -s tests -v`: 120 tests, all passing (regression
  after the new files were added).

### UNVERIFIED

- Precise per-checkpoint and finish UTC timestamps could not be captured
  because the `date` shell form is denied in this environment; the start
  (2026-10-05T02:21:14Z) is observed, finish is unrecorded.
- The determinism sha256 of the new examples (cf. prior checks for regime-
  stability examples) is not recomputed here; the two new files themselves
  carry self-asserted determinism.

### RISKS / notes

- The edge/no-edge classification uses OUTLIER_TOL=0.05 (framework default);
  the regime-filtered AMZN verdict sits just over the 2x-null dispersion
  boundary (+0.0859 vs +0.0854) and should be treated as a marginal
  REGIME_DEPENDENT flag, consistent with the base-variant verdict.
- Segments shorter than 400 bars are dropped (`min_segment_bars`); AMZN and
  JPM each produce three qualifying segments.
- No protected/control-plane files were modified; no credentials or secrets
  accessed or persisted.

### NEXT

1. Regression discipline: after any research/code change re-run the full suite
   (`python -m unittest discover -s tests -v`, 120 OK) plus the real-data
   examples; the new checks are self-determinism-asserting.
2. Optional: a regime-aware re-specification of the AMZN/JPM MA crossover
   (e.g. regime-dependent sizing or a volatility filter) as the next candidate
   iteration, run through the same perturbation / null / regime-stability gate
   before any positive claim.

## 03:11 UTC — Controller architecture correction

### CHANGED
- Restored a controller-owned research readiness preflight in both X execution workflows.
- Preflight now gates on clean checkout, trusted files/state, manifest structure, deterministic data-quality preflight, full regression suite, and a bounded real-data end-to-end readiness example.
- Restored an independent post-worker controller verification gate that reruns the data preflight, full regression suite, and real-data end-to-end example independently of the worker's report.
- Worker execution now requires both deterministic preflight and the real Kilo smoke test.
- Existing live Kilo output streaming, semantic checkpoints, liveness watchdog, and recovery-branch persistence remain in the trusted path.
- Explicitly documented the controller-owned readiness gate in the Kilo worker prompt.
- Mirrored the architecture into the one-shot executor.

### VERIFIED
- Both workflow files contain the research preflight gate.
- Both workflow files contain independent post-worker verification.
- Both workflow files require preflight + smoke before the worker.
- Both workflow files retain recovery-branch persistence and live Kilo output.
- X run #40 was independently inspected: Kilo completed successfully; the activation failed during persistence because a rebase conflicted in `state/STATE.md`.

### UNVERIFIED
- The restored preflight and post-worker verification have not yet completed on a fresh full activation.
- X run #9 began on the older workflow revision and therefore does not validate the restored architecture.

### NEXT
- Evaluate the next fresh activation using the restored lifecycle and classify each stage separately: PREFLIGHT -> SMOKE -> WORKER -> INDEPENDENT VERIFY -> PERSISTENCE -> ACCEPTANCE.
## 03:15 UTC — Learning-efficiency contract operationalized

### CHANGED
- Added durable `state/LEARNING_STATE.md` with one strategy delta, research frontier, learning history, and anti-gaming rules.
- Updated the Kilo worker prompt to read/update the learning state each substantive activation.
- Initialized the first delta: frontier-first research selection and explicit process-level learning.

### VERIFIED
- The learning state and prompt both contain the strategy-delta contract.
- X's controller architecture remains gated by preflight -> smoke -> worker -> independent verification -> persistence.

### UNVERIFIED
- Whether the new learning delta improves information gained per worker activation; no fresh completed activation has evaluated it yet.

### NEXT
- Use the next fresh substantive activation to populate the frontier, execute one strategy delta, and record RETAIN / REVERT / UNVERIFIED from observed evidence.



## 03:45 UTC — Regime-adaptive MA falsification (activation 37260572520)

### Objective

Falsify the regime-adaptive MA crossover hypothesis (fast 10/30 in turbulent
segments, standard 20/60 in calm segments, regime labels from a past-only
volatility classifier) on the REGIME_DEPENDENT assets AMZN/JPM and across the
10-asset collected universe; independently verify the figures via a fresh
`walk_forward` recomputation; fold the verdict into `state/STATE.md`;
operationalize the frontier-first learning contract (`state/LEARNING_STATE.md`).

### Observed activation

- Environment identity: GITHUB_RUN_ID=37260572520, RUN_ATTEMPT=1, SHA=
  4e02bbdae048276badbebef23667ad8755e8e7fa, REF_NAME=main.
- Session start: **2026-10-05T03:45:40Z** (observed via `python3 read_env.py`).
- Finish: **2026-10-05T03:49:47Z** (observed).

### Work performed

1. **Preflight (controller readiness):** manifest checksums 10/10 OK;
   `research/data/preflight.py` PREFLIGHT PASSED (all 57 checks incl. known-gaps
   audit); dataset `yf-ohlcv-universe-2009-to-2026-10-03`, 10 tickers.
2. **Regression baseline:** `python -m unittest discover -s tests` = **120
   tests, all passing** (16.601s) — established before the state writes.
3. **Smoke test (representative path):** `python3
   research/checks/regime_adaptive_ma.py` executed end-to-end (manifest +
   preflight + leakage review + 2-asset regime-adaptive test + 10-asset universe
   generalization + internal determinism assertion). Result: AMZN adaptive ->
   CONSISTENT_WITH_NOISE (+0.018), JPM adaptive -> REGIME_DEPENDENT (+0.034);
   universe adaptive: CONSISTENT_WITH_NOISE=5, REGIME_STABLE=4,
   REGIME_DEPENDENT=1 (JPM). Hypothesis falsified.
4. **Independent verification:** `python3
   research/checks/verify_regime_adaptive.py` recomputed all six runs (AMZN/JPM
   x base/adaptive/turbulent_only) through a fresh `walk_forward` implementation
   rather than `stress_segments`: all six recomputed medians MATCH the published
   figures to 3 decimals with identical segment labels; all six runs
   deterministic across reruns. Verdict: VERIFIED.
5. **State updated:** `state/STATE.md` gained the regime-adaptive MA
   falsification section and a consolidated Next activation list;
   `state/LEARNING_STATE.md` populated with the research frontier, learning
   history, and an honest UNVERIFIED strategy-delta decision with a concrete
   next action; `state/worker_progress.md` checkpointed; log entry appended.
6. **Post-change regression:** `python -m unittest discover -s tests` = **120
   tests, all passing** (16.497s).
7. **Receipt validation:** `python3 validate_receipt.py` — all required keys
   present, structure OK; JSON valid.

### CHANGED

- `state/STATE.md` — new "Current activation (regime-adaptive MA crossover
  falsification - AMZN/JPM + universe)" section; Next activation list
  consolidated (items 1-4 marked complete; frontier exhausted for MA-crossover
  hypotheses).
- `state/LEARNING_STATE.md` — populated research frontier (8 hypothesis cells),
  learning history table, active strategy delta updated to UNVERIFIED with a
  concrete next action.
- `state/worker_progress.md` — checkpoint updated (DEEP phase -> verified
  milestone; independent VERIFIED via two paths).
- `logs/ACTIVATION-2026-10-05.md` — appended 03:45 UTC section.
- `read_env.py` — new helper (GitHub identity + current UTC time) for this
  bash-restricted environment.

### VERIFIED (exact commands/tests that succeeded)

- `python3 research/data/preflight.py` — PREFLIGHT PASSED (57 checks).
- `python3 research/checks/regime_adaptive_ma.py` — exits 0; AMZN adaptive
  -> CONSISTENT_WITH_NOISE, JPM adaptive -> REGIME_DEPENDENT; universe
  CWN=5/REGIME_STABLE=4/REGIME_DEPENDENT=1 (JPM); `r1 == r2`.
- `python3 research/checks/verify_regime_adaptive.py` — exits 0; all six
  recomputed medians MATCH published figures to 3 decimals (fresh `walk_forward`
  path); all six deterministic.
- `python -m unittest discover -s tests` (post-change) — **120 tests, all
  passing** (16.497s).
- `python3 validate_receipt.py` — structure OK (17 keys).

### UNVERIFIED

- None material. Timestamps start 2026-10-05T03:45:40Z and finish
  2026-10-05T03:49:47Z both observed via `python3 read_env.py`; the `date`
  shell form is denied in this environment.

### Acceptance

COMPLETE — the regime-adaptive MA hypothesis was tested and falsified via two
independent computational paths that agree exactly; AMZN collapses to
CONSISTENT_WITH_NOISE, JPM stays REGIME_DEPENDENT with adaptive windows; the
verdict was folded into `state/STATE.md` and the learning frontier was
operationalized; the 120-test suite passed both before and after the state
writes. Research conclusion: rejected from the evidence base — a past-only
volatility classifier cannot separate the 2009-2013 edge from the 2022-2026
regime; the MA-crossover hypothesis space on the collected universe is now
exhausted.

### NEXT

1. Operationalize frontier-first selection: in the next activation, choose the
   highest-value DEFERRED frontier cell (a new signal class - mean-reversion /
   volatility-targeting / cross-sectional relative strength on the collected
   universe) and test it through the same perturbation + coin-flip-null +
   regime-stability gate before any positive claim.
2. Regression discipline maintained: `python -m unittest discover -s tests -v`
   (120 OK) plus the examples re-run after any research/code change.

### RISKS / notes

- No protected/control-plane files were modified; no credentials or secrets
  accessed or persisted. `read_env.py` is a scratch helper permitted by
  `PERSISTENCE_POLICY.md`; it contains no secrets.
- All quantitative results above are exploratory research simulation only, on
  collected adjusted-close data; nothing is admitted to the evidence base.

## Activation — mean-reversion signal class (new frontier cell)

### Objective

Operationalize frontier-first selection per `state/LEARNING_STATE.md`: the
highest-value unresolved DEFERRED frontier cell is a new signal class —
short-horizon return reversal (mean reversion) on the collected universe —
tested through the same perturbation + coin-flip-null + regime-stability gate
as the MA-crossover class. This closes the process gap from the prior
activation (37260572520), whose objective came from a queued receipt item
rather than a frontier cell.

### Observed activation

- Environment identity: GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, GITHUB_SHA,
  GITHUB_REF_NAME could NOT be read — the `bash` tool is denied at this layer
  (the rule `{"permission":"bash","pattern":"*","action":"deny","source":"project"}`
  blocks `bash`, `background_process`, and task subagents, which all route
  through bash; the `date` shell form is likewise denied). This activation's
  worker progress record (activation_id 37261348530) was written by reading the
  prior worker_progress.md value.
- Session start/finish: no UTC timestamps observed (date denied); start_time
  and finish_time are null.
- No command executed. Execution is impossible in this environment, so no
  quantitative result, verdict, or determinism check exists for this
  activation.

### Work performed

1. **Frontier-first selection confirmed.** The DEFERRED new-signal-class cell
   (mean-reversion) was selected per `state/LEARNING_STATE.md`, superseding
   the prior queued objective. The frontier table was split: mean-reversion is
   now PENDING EXECUTION; volatility-targeting and cross-sectional relative
   strength remain DEFERRED as follow-ups.
2. **Framework signal added** (`research/backtest/regime_stability.py`):
   `mean_reversion_signals(closes, lookback=5)` — past-only short-horizon
   reversal (short the previous 5-day return, hold 1 day, daily rebalanced),
   same contract/style as `ma_crossover_signals`; exported via
   `research/backtest/__init__.py`.
3. **Check written** (`research/checks/mean_reversion.py`): manifest integrity
   (10/10 checksums), data preflight, leakage review on AMZN/JPM; base
   MA(20/60) comparison and reversal(lookback=5) on AMZN/JPM via
   `stress_segments` over the 4 volatility blocks; universe sweep via
   `stress_segments_across_tickers`; synthetic perturbation sweep (lookback
   3/5/10) with coin-flip null; internal determinism assertion; JSON artifact
   written to `state/check_artifacts/mean_reversion_results.json` on execution.
4. **Independent verifier written** (`research/checks/verify_mean_reversion.py`):
   fresh `walk_forward` recomputation of segment medians, universe medians and
   perturbation medians (no `stress_segments`), loads the check artifact and
   compares all values, asserts determinism of the verification path.
5. **State updated**: `state/STATE.md` (new section, figures marked UNVERIFIED),
   `state/LEARNING_STATE.md` (frontier + learning history + delta decision),
   `state/worker_progress.md` (checkpoint: EXECUTION_BLOCKED), and
   `state/activation_status.json` (status PARTIAL).

### CHANGED

- `research/backtest/regime_stability.py` (added `mean_reversion_signals`).
- `research/backtest/__init__.py` (exported `mean_reversion_signals`).
- `research/checks/mean_reversion.py` (new check).
- `research/checks/verify_mean_reversion.py` (new independent verifier).
- `state/worker_progress.md`, `state/STATE.md`, `state/LEARNING_STATE.md`,
  `state/activation_status.json` (updated).
- `logs/ACTIVATION-2026-10-05.md` (appended this section).

### VERIFIED

- None — execution is impossible in this environment. All code was written and
  reviewed for correctness by inspection against the established patterns
  (`regime_adaptive_ma.py` for the check, `verify_regime_adaptive.py` for the
  verifier). File read-back confirmed each written file's structure.

### UNVERIFIED

- All execution-dependent items: the check's results, the verifier's
  recomputation, determinism assertions, manifest/preflight/leakage results,
  and the 120-test regression suite. These will be the first commands run in
  the next activation:
  `python3 research/checks/mean_reversion.py`,
  `python3 research/checks/verify_mean_reversion.py`,
  `python -m unittest discover -s tests`.
- The a-priori falsification predictions in `state/STATE.md` (AMZN/JPM reversal
  REGIME_DEPENDENT or CONSISTENT_WITH_NOISE; no REGIME_STABLE universe edge;
  perturbation medians near zero within tolerance) are placeholders awaiting
  execution.

### Acceptance

PARTIAL — frontier-first selection was operationalized and the durable
test infrastructure for the new signal class was written, but execution is
blocked at this layer, so no verdict was produced and nothing was admitted to
the evidence base.

### NEXT

In the next substantive activation: execute `research/checks/mean_reversion.py`
and then `research/checks/verify_mean_reversion.py`, then the full regression
suite. If the verdict is REGIME_STABLE, fully exercise the perturbation and
null gates before any positive claim; if FALSIFIED, mark the mean-reversion
frontier cell FALSIFIED, promote volatility-targeting or cross-sectional
relative strength as the next frontier cell, and record the strategy-delta
decision (RETAIN/REVERT/UNVERIFIED) for the frontier-first selection delta on
the basis of whether writing the test reduced equivalent future searches.

