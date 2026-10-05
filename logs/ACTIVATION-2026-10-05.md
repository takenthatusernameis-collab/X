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

