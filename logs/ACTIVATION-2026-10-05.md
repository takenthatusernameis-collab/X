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
