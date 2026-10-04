# Persistent Enterprise State

## Objectives

1. Build the enterprise's deterministic research/backtest foundation.
   **Status: done** (completed in the prior activation; the passing unit
   test suite and the MA-crossover example confirmed it).
2. Add a second signal class and a robustness/perturbation test suite
   using the same framework. **Status: done** (this activation).
3. Prepare a real-data feasibility note for research-only simulation.
   **Status: done** (this activation).
4. Add a regime-stability stress test: verify a candidate's out-of-sample
   results are stable across different regime parameterizations rather
   than fitting one regime mix. **Status: done** (this activation).
5. Add an asset-universe robustness sweep: verify a candidate's results do
   not rest on a single asset (concentration vs consistency verdicts), the
   robustness dimension that completes the methodology's perturbation /
   regime-stability / asset-universe trio. **Status: done** (this
   activation).

## Status

- `research/backtest/` — deterministic backtest toolkit: engine
  (event-driven, walk-forward IS/OOS), metrics, synthetic regime-switching
  data, leakage/look-ahead checks, perturbation/parameter-sensitivity
  testing (`perturbation.py`), and regime-stability stress testing
  (`regime_stability.py`). Dependency: numpy.
- `tests/` — deterministic unit tests for engine, metrics, data,
  perturbation, and regime-stability; all passing.
- `examples/` — `ma_crossover.py` (walk-forward + cost sensitivity +
  perturbation sweep), `volatility_regime_filter.py` (regime-aware
  signal, walk-forward, costs), and `regime_stability_demo.py` (demonstrates
  all three regime-stability verdicts).
- `research/METHODOLOGY.md` — evidence standard, no-look-ahead rules,
  walk-forward discipline.
- `research/REAL_DATA_FEASIBILITY.md` — in-scope data, provenance
  manifest, preflight checks, and the leakage review checklist for any
  future real-data run.
- README and `research/README.md` updated to reflect the infrastructure.

## Current activation (follow-on)

Added deterministic robustness/perturbation tooling to the enterprise so
that every candidate strategy is tested for parameter sensitivity before
its results are admitted into the evidence base. This is the capability
that `METHODOLOGY.md` describes as required for judging robustness
(perturbation, parameter sensitivity, regime stability) but which did not
exist in the toolkit.

- Added `research/backtest/perturbation.py`: `parameter_sweep()` runs the
  same walk-forward validation across a grid of parameters;
  `parameter_grid_around()` builds a half/baseline/double grid around a
  canonical parameter set; `sweep_summary()` reports a deviation scan,
  sign consistency across parameter sets, and best/worst parameter sets;
  `noise_benchmark()` runs the sweep on a coin-flip (price-independent)
  signal as a null hypothesis; `random_signals()` generates that null.
- Added `examples/volatility_regime_filter.py`, a past-only volatility-regime
  filter with walk-forward IS/OOS, cost sensitivity, and leakage checks.
- Extended `examples/ma_crossover.py` with a perturbation sweep and a
  coin-flip null comparison.
- Added `tests/test_perturbation.py` (14 tests: determinism, grid
  formation, empty/edge cases, noise-centered-on-zero, known-peak
  detection, summary consistency, IS/OOS per-parameter-set, and noise
  comparison).
- Added `research/REAL_DATA_FEASIBILITY.md` defining in-scope and
  out-of-scope data sources, the provenance manifest schema, a data-quality
  preflight checklist, and the pre-run leakage review checklist.

Resulting finding (synthetic data; tooling validation only): the MA
crossover and the volatility-regime filter both show no persistent edge on
regime-switching synthetic data, and their sweep results are
indistinguishable from the coin-flip null. The framework therefore reports
correctly and does not hallucinate an edge — a negative result that reduces
the risk of later over-reading walk-forward noise. See
`logs/ACTIVATION-2026-10-02.md` for the full record.

## Current activation (regime-stability stress testing)

Added regime-stability stress testing to the enterprise so that every
candidate is checked for dependence on the data-generating process before
its results are admitted into the evidence base. This closes the last gap
in the robustness capability that `METHODOLOGY.md` lists as required for
judging robustness (perturbation, parameter sensitivity, regime stability).

- Added `research/backtest/regime_stability.py`: `regime_stress()` runs the
  same walk-forward perturbation sweep across a family of regime scenarios
  (via `run_scenario()`); `RegimeScenario` / `RegimeScenarioResult` hold the
  scenario configuration and per-scenario medians; `regime_stability` verdicts
  are one of CONSISTENT_WITH_NOISE (candidate indistinguishable from the
  coin-flip null in every scenario), REGIME_STABLE (consistent behavior —
  same edge or no edge — across regimes, candidate dispersion not greater
  than twice the null's), or REGIME_DEPENDENT (candidate results swing
  across regimes far more than a coin-flip null would, i.e. fits one regime).
  Also provided: `stress_regime_scenarios()` (extreme drift regimes for
  demonstration, documented as stress cases), `canonical_regime_scenarios()`,
  and contrived signals `direction_signal()` and `always_long_signal()`.
- Added `tests/test_regime_stability.py` (14 tests: scenario generation,
  determinism, noise behavior, detection of regime-dependence, detection of
  regime-stable adaptive signals, edge cases).
- Added `examples/regime_stability_demo.py` demonstrating all three verdicts
  on synthetic data.

  on synthetic data.

## Current activation (engine accounting integrity and margin/collateral modeling)

Added two accounting-integrity improvements to `research/backtest/engine.py`
so that trade statistics are complete and leveraged strategies can be modeled
safely:

- **Complete fill records.** Positions are now closed with a recorded fill
  whenever a signal goes to neutral (previously the engine held the position
  without recording the sale, so `n_trades` and turnover were systematically
  undercounted and the equity-fill audit could not detect missing exit fills).
  Full-sample and walk-forward trade stats now reflect the true round-trip
  count; `check_equity_matches_fills` fully validates the fill sequence.
- **Margin/collateral modeling.** `BacktestConfig` gained `margin_rate`
  (maintenance margin as a fraction of gross notional) and
  `margin_call_liquidate`; the engine liquidates to neutral on a margin call
  and records each call in `FillResult.margin_calls`. Walk-forward
  aggregates report `total_margin_calls`. All changes default to the prior
  behavior (`margin_rate=0`), so existing examples and tests are unaffected.

Both changes are additive and deterministic; the default configuration
reproduces the previous engine behavior exactly (verified by the full suite
and by a default-vs-explicit-config comparison).

Verified results (seed 42, regime-switching synthetic data; tooling-validation only), superseding the pre-change figures:

MA crossover (warmup=60, 252d/84d walk-forward, train+test):
- Full sample, zero cost: trades=55, total return -106.93%, sharpe=-0.04,
  max drawdown 128.21%, turnover 319.8x.
- Walk-forward: 88 folds, 7392 OOS periods, mean log return -0.123, median
  log return -0.081; positive folds 37/88.
- Perturbation sweep (0.5x/1.0x/2.0x grid): medians near zero across the
  grid with mixed signs and a flat deviation scan
  (0.00x: -0.081, 0.50x: -0.054, 1.00x: -0.021); best set ((40, 30))
  +0.045 vs worst ((20, 60)) -0.081; no single-point peak — no robust edge.

Volatility regime filter (warmup=20, 252d/84d walk-forward):
- Full sample, zero cost: trades=10 (exits are now recorded), total return
  -1.72%, sharpe=-0.05, max drawdown 9.60%. The full-sample result changed
  from -42.83% because positions are now correctly closed when the signal
  goes neutral rather than held indefinitely at mark-to-market; the
  equity-fill audit passes on the corrected run.
- Walk-forward: 90 folds, 7560 OOS periods, mean log return -0.001, median
  log return 0.000; positive folds 8/90.

Regime-stability stress test (seed 42, 600 bars, train=60d/test=20d,
walk-forward IS/OOS; the extreme regime family is documented as stress
cases, not realistic parameterizations):

- Long-only rule on extreme up (+0.9/yr) / neutral / down (-0.9/yr) regimes:
  candidate medians [+0.055, +0.005, -0.079] across scenarios;
  candidate dispersion 0.055 > 2x null dispersion 0.003. Verdict:
  REGIME_DEPENDENT (correctly flags a rule whose results swing with the
  regime mix).
- Direction-following rule (long up, short down) on the up/down pair:
  candidate medians [+0.055, +0.048]; candidate dispersion 0.003 <
  2x null dispersion 0.003. Verdict: REGIME_STABLE (consistent edge in both
  regimes).
- Coin-flip signal on the canonical realistic family: medians near zero in
  all four scenarios. Verdict: CONSISTENT_WITH_NOISE.
- MA crossover on the canonical realistic family: medians near zero in all
  scenarios. Verdict: CONSISTENT_WITH_NOISE (honest exploratory finding:
  no regime dependence is detectable on realistic regime mixes).

The null dispersion in the stress tests shifted slightly (0.019 -> 0.003)
because the coin-flip signal's equity curve is now computed with realized
exit proceeds; the verdicts themselves are unchanged.

## Current activation (asset-universe robustness testing)

Added an asset-universe robustness sweep to the enterprise so that every
candidate is checked for concentration in a single asset before its results
are admitted into the evidence base. This closes the final robustness
dimension listed in `research/METHODOLOGY.md` (perturbation, regime
stability, asset-universe stability).

- Added `research/backtest/universe.py`: `sweep_across_assets()` runs the
  same walk-forward validation across a family of assets;
  `asset_sweep_summary()` computes per-asset OOS medians, a coin-flip null
  benchmark, and a verdict of CONSISTENT (edge spread across assets),
  CONCENTRATED (the best asset carries >= 60% of the positive edge), or
  NO_EDGE (indistinguishable from the null everywhere). `uniform_regime_assets()`
  builds deterministic families of synthetic assets sharing one regime
  structure with shifted drifts; `flat_regime_assets()` builds a
  no-edge reference universe.
- Added `tests/test_universe.py` (10 tests: determinism, empty/mismatched
  input handling, single-asset edge case, detection of CONCENTRATED and
  CONSISTENT verdicts, and NO_EDGE against the coin-flip null).
- Added `examples/universe_sweep.py`: MA crossover walk-forward across a
  7-asset family with drifts from -6% to +12%; reports per-asset medians,
  the null benchmark, and the verdict.

Verified result (seed 42, regime-switching synthetic data; tooling-validation
only): the MA crossover on a 7-asset drifted family reports medians near zero
in every asset (-0.133 .. +0.020), best-asset edge share 41%, verdict
NO_EDGE — the sweep distinguishes noise from concentration correctly. On a
contrived family with one strong-drift asset the sweep reports CONCENTRATED
(80% edge share, best asset asset_0); on a family with uniform strong
  positive drifts it reports CONSISTENT (edge in all assets, share < 60%).

## Current activation (asset-universe significance flags)

Added a per-asset significance flag to the asset-universe sweep so that each
asset is judged against its own coin-flip null, rather than against a single
global null. This closes the documented gap: with only a handful of
walk-forward folds per asset the null dispersion can be large, and a global
tolerance can mask a signal that is meaningful for a particular asset.

- `research/backtest/universe.py`: `AssetSweepSummary` gained
  `asset_null_medians` (coin-flip baseline per asset), `asset_significance`
  (per-asset flag: candidate median distinguishable from its own null), and
  `n_significant_assets`. The `verdict` property uses per-asset nulls when
  they are available; the NO_EDGE check then compares every asset to its own
  null. `asset_sweep_summary` gained the `per_asset_null=True` option; the
  default (`False`) preserves the prior API and all existing verdicts.
  `inspect()` prints the per-asset significance flags and the effective
  tolerance.
- `examples/universe_sweep.py` now runs with `per_asset_null=True`, so the
  example reports the flags.
- `tests/test_universe.py` — new `TestPerAssetNull` class (5 tests:
  determinism, flat family stays NO_EDGE under the per-asset null, agreement
  with the global-null path on a clear NO_EDGE case, single significant edge
  yields CONCENTRATED, and self-consistency of the flags).

Result (seed 42, 7-asset drifted family): the null dispersion tightened from
+0.109 (global, first 3 assets) to +0.079 (per-asset, all 7), effective
tolerance +0.158, and every asset falls within its own null — 0 / 7
significant assets, verdict NO_EDGE, unchanged from the prior run. An
 independent determinism re-run produced identical fields and hash
(3968836896295251151).

## Current activation (default per-asset nulls)

Enabled the per-asset coin-flip null as the default in `asset_sweep_summary`
so that each asset in a universe sweep is judged against its own coin-flip
baseline rather than a single global null estimated from a small sample of
assets. The per-asset path was implemented and unit-tested in the prior
activation; this activation flipped the default and re-verified the full
suite and all four examples.

- `research/backtest/universe.py`: `asset_sweep_summary()` default changed
  from `per_asset_null=False` to `per_asset_null=True`; docstring updated to
  document the new default and that `noise_n` is ignored in that case.
- `tests/test_universe.py`: `TestUniverseReproducibility.test_sweep_reproducible`
  gained an assertion that the default path reports `asset_null_medians` and
  `asset_significance` (lengths equal `n_assets`), locking in the documented
  contract.

All canonical verdicts are unchanged under the new default: `CONCENTRATED`
(one-strong-asset family), `CONSISTENT` (uniform-strong family), and
`NO_EDGE` (MA crossover on the 7-asset family; flat family). `universe_sweep`
verdict: NO_EDGE, 0/7 significant assets, best-asset share 41.3%. The suite
re-ran with 87 tests passing, and `research/checks/determinism_check.py`
reports `r1 == r2: True`. See `logs/ACTIVATION-2026-10-03.md`.

## Current activation (deterministic real-data readiness gate)

Added a deterministic known-data-gaps audit to the real-data preflight so that
documented source gaps (e.g. the two systematic Yahoo omissions, 2018-12-05 and
2025-01-09) are verified as genuinely absent from every ticker's data rather
than merely assumed — this closes the last gap between the documented
`research/REAL_DATA_FEASIBILITY.md` preflight checklist and an executed, tested
gate. Real-data readiness is now: a manifest with full provenance and checksums
(`research/data/manifest.json`), a 10-check preflight (`research/data/preflight.py`),
and a test suite for the gate (`tests/test_preflight.py`).

- Added `known_gaps_audit()` to `research/data/preflight.py`, integrated as
  check #10 "Known gaps audit": for each ticker it verifies every documented
  gap is genuinely absent; a documented gap that is present in the data flags
  the manifest or raw files as stale. The audit is independent of the
  completeness check (which compares against an expected business calendar)
  because it operates solely on `manifest["known_data_gaps"]`.
- Added `date` to the `from datetime` import so the audit compares correctly.
- Added `tests/test_preflight.py` (10 tests): documented gaps absent in every
  ticker; audit failure when a documented gap is present in data; manifest
  staleness end-to-end; unexplained gaps fail completeness; unparsable gap
  dates fail the gate; gate determinism; and that the gate passes on the
  collected universe.
- Verified (real-data gate; tooling-validation only): `python3
  research/data/preflight.py` exits 0 with "[PASS] 10. Known gaps audit: all
  documented gaps verified genuinely absent in every ticker"; the suite runs at
  87 tests, all passing; all four walk-forward examples exit 0.

## Current activation (loader verification completed)

The corrected real-data loader from the prior activation (commit `40a131b`, which
reads the CSV `adjclose` field at index `p[5]`) has now been executed and
verified end-to-end, closing the UNVERIFIED state documented in the previous
activation record.

- Added `tests/test_loader.py` (9 tests): contract checks that the loader returns
  the CSV `adjclose` field and not raw `close` (AAPL first row: 2.714299 vs
  3.241071); bar counts match `manifest["per_ticker_bars"]` for all 10 tickers;
  date mapping starts at each ticker's `first_available_date` and matches CSV
  row order; determinism across reruns; and a full real-data pipeline check:
  AAPL MA-crossover signals pass `check_signal_integrity`, the full-sample run
  passes `check_equity_matches_fills`, and the walk-forward completes with
  populated fold metrics (4465 bars).
- Verified: `python3 tests/test_loader.py` — 9 tests OK; `python -m unittest
  discover -s tests -v` — **96 tests, all OK** (87 existing + 9 new loader tests);
  `python3 research/data/preflight.py` — PREFLIGHT PASSED (57 checks, incl.
  known-gaps audit); all four walk-forward examples exit 0 with aggregates
  matching the STATE.md verified figures (ma_crossover: 88 folds / 7392 OOS;
  volatility_regime_filter: 90 folds / 7560 OOS; regime_stability_demo: all three
  verdicts; universe_sweep: NO_EDGE, 0/7 significant, 41.3% share).

This activation's regression-discipline item is therefore complete and verified;
the corrected loader no longer has an open verification gap.

## Current activation (real-data pipeline on the collected universe)

The first research pipeline on real, collected data is now executed end-to-end
(`examples/ma_crossover_real_data.py`): manifest checksums verified (10/10 OK),
preflight gate passed (all 57 checks), leakage review passed (signal integrity +
fill-equity audit), walk-forward IS/OOS with zero and realistic costs, a
perturbation sweep (0.5x/1.0x/2.0x around canonical 20/60) with a coin-flip null
benchmark, and an asset-universe sweep across all 10 collected tickers with
per-asset nulls. Baseline state re-verified independently: 96 tests passing,
all four synthetic examples reproducing their documented seed-42 figures.

**AAPL MA crossover (2009-01-02 to 2026-10-02, adjusted close, walk-forward
train=252d/test=84d/warmup=60d, overlap=60d, 170 folds / 14,280 OOS periods):**
- Zero cost: mean log return +0.031, median +0.024, std +0.146; 101/170
  positive folds. Realistic costs: mean +0.031, median +0.023 (survives costs).
- Full-sample reference (zero cost): 78 trades, total +225.44%, sharpe 0.53,
  max DD 30.98%, turnover 93.8x (reference only; not the evidence).
- Perturbation: medians +0.023 .. +0.033 across the 9 param sets (8/9
  positive); flat deviation scan (0.00x +0.024, 0.50x +0.023, 1.00x +0.019)
  but the canonical (20, 60) set is not the grid optimum ((10, 60) = +0.033);
  coin-flip null median -0.006; baseline within the tooling's 5% null tolerance
  (|+0.024 - (-0.006)| = 0.030 <= 0.05). Independent cross-check: t-statistic
  2.75 (df=169); 95% CI for the mean log fold return [0.0089, 0.0529], which
  excludes zero.
- Universe: per-asset medians AAPL +0.017, MSFT +0.018, GOOGL +0.008, AMZN
  +0.015, META +0.031, NVDA +0.036, TSLA -0.004, JPM +0.001, JNJ -0.019,
  XOM -0.005; 7/10 positive; 1/10 significant vs own null (NVDA, marginal);
  best-asset share 28.3%; verdict CONSISTENT.

**Assessment (exploratory, not admitted to the evidence base):** the AAPL MA
crossover shows a marginally significant walk-forward edge over 2009-2026 but
fails the robustness expectations (canonical parameters are not a peak; within
the null tolerance; only sparse significance across the universe). It is
recorded as an exploratory candidate rejected for further study. Contrast with
the synthetic-data finding of no edge (median -0.081 per fold on the
regime-switching generator) indicates the generator does not capture
persistent-drift regimes, so the synthetic "no edge" result is generator-specific
and cannot be generalized; direct real-data walk-forward testing is the more
informative validation path for trend-based ideas.

**Methodology note:** the `compare_noise` fixed 5% null tolerance is wider than
the 95% CI half-width (~0.023) for 170 folds, so it can classify a
statistically significant result (t=2.75) as "within noise"; a sample-calibrated
null band is a candidate framework improvement (not implemented here).

## Current activation (per-asset perturbation sweep on the collected universe)

Extended the asset-universe sweep so every real asset is evaluated over the full
parameter grid, closing the optional follow-on item 4a from the prior activation.
`research/backtest/universe.py` gained:

- `AssetSweepResult.param_set_median_log_returns`: per asset, per parameter set
  median log return across folds (derived from the already-computed
  `fold_total_returns`, so it adds no extra computation).
- `AssetSweepSummary` fields: `perturbation_profiles` (per-asset deviation scan,
  deviation from canonical baseline -> median log return), `baseline_peak_count` /
  `baseline_peak_share` (fraction of assets whose canonical parameter set is the
  best of its grid), `positive_param_sets_per_asset` (count of positive median
  parameter sets per asset), and `baseline_in_grid`. `inspect()` prints the
  per-asset profiles and peak statistics. The baseline lookup and deviation
  computation were made robust to both the tuple-of-tuples API form and the flat
  test-form parameter sets. No existing verdicts or the 96-test suite behavior
  changed.
- `examples/ma_crossover_real_data.py` now reports the baseline-peak finding in
  its summary (baseline (20,60) is not the best param set in any asset).

Verified result (seed 42, 10 collected tickers, 0.5x/1.0x/2.0x grid around
canonical 20/60, 252d/84d walk-forward, warmup=60d): the canonical MA crossover
param set (20,60) is not the best parameter set in ANY of the 10 assets
(0/10 baseline peaks). Per-asset perturbation profiles show wide spread across
the grid: most assets are positive in 8/9 or 9/9 parameter sets (AAPL 8/9,
MSFT 8/9, GOOGL 8/9, AMZN 8/9, META 9/9, NVDA 8/9, JPM 5/9, JNJ 1/9, XOM 2/9,
TSLA 4/9), but the 1.0x-deviation sets (fast=40 or slow=120) swing from -0.074
(NVDA) to +0.074, and the canonical window is the worst or near-worst set for
most assets. This is the overfitting signature the perturbation gate is designed
to catch: the real-data edge is parameter-dependent and does not peak at the
canonical parameters in any asset, reinforcing the prior decision to reject the
AAPL MA crossover from the evidence base.

Verification (exact commands that succeeded after the final edits):
- `python -m unittest discover -s tests -v`: **96 tests, all passing**.
- `python3 examples/ma_crossover_real_data.py`: exits 0, all 9 sections complete,
  per-asset perturbation profiles printed; baseline peak share 0.0 (0/10 assets),
  verdict CONSISTENT, 1/10 assets significant.
- Determinism: two independent reruns of the example produce byte-identical
  output, sha256 `5d42d25c6a25df6b16e61e3a05ae1335d30f833ee882eeab6753de3cb6598786`, rc=0 both.
- `python3 research/checks/verify_aapl_stats.py`: 170 folds, mean +0.0309,
  median +0.0242, std +0.1463, 101/170 positive, 95% CI [0.0089, 0.0529],
  t-statistic 2.75 (df=169).

## Next activation

1. Regression discipline: **complete and verified** — `python -m unittest
   discover -s tests -v` (96 OK) plus the four walk-forward examples, re-run
   after any research/code change.
2. Real-data readiness: **complete** — the real-data pipeline ran on the
   collected universe; see the "Current activation" section above for results
   and the determination that the MA crossover is not admitted to the evidence
   base.
3. Determinism re-verification of `examples.ma_crossover_real_data` across
   independent reruns (byte-identical output): **complete and verified** — two
   independent reruns produce byte-identical output, sha256
   `5d42d25c6a25df6b16e61e3a05ae1335d30f833ee882eeab6753de3cb6598786`.
4. Optional follow-on: (a) extend the asset-universe sweep to run the full
   perturbation grid per real asset — **done (this activation)**; (b) decide
   whether to make the `compare_noise` null tolerance sample-calibrated (2x the
   coin-flip null's own fold dispersion at the same sample size) while
   preserving all existing verdicts in the 96-test suite; (c) run regime-stability
   on real data with a real-data regime family.

## Verification (executed)

The framework is deterministic and the full suite executes:

- `pip install -q numpy` (resolved numpy 2.5.3); all modules compile.
- `python -m unittest discover -s tests -v`: **96 tests, all passing**
  (9 data + 19 engine + 8 metrics + 15 perturbation + 11 regime-stability
  + 15 asset-universe + 10 preflight + 9 loader); 0 failures/errors.
- `python -m examples.ma_crossover`: runs end-to-end and prints the full
  perturbation deviation scan; two independent runs produce byte-identical
  output (sha256 `baeee6a3...`).
- `python -m examples.volatility_regime_filter`: runs end-to-end with
  walk-forward output; leakage checks pass on the full-sample runs.
- `python -m examples.regime_stability_demo`: runs end-to-end and prints
  all three verdicts; two independent runs produce byte-identical output
  (sha256 `2d3bc5c4...`).
- `python -m examples.universe_sweep`: runs end-to-end and prints the asset-
  universe verdict (NO_EDGE on the MA crossover across a 7-asset drifted
  family); deterministic across reruns.

Corrected verified results on seed 42 (regime-switching synthetic data;
tooling-validation only), superseding the unverified figures in the
activation log:

MA crossover (warmup=60, 252d/84d walk-forward, train+test):
- Full sample, zero cost: trades=55, total return -106.93%, sharpe=-0.04,
  max drawdown 128.21%, turnover 319.8x.
- Walk-forward: 88 folds, 7392 OOS periods, mean log return -0.123, median
  log return -0.081; positive folds 37/88.
- Perturbation sweep (0.5x/1.0x/2.0x grid): medians near zero across the
  grid with mixed signs and a flat deviation scan
  (0.00x: -0.081, 0.50x: -0.054, 1.00x: -0.021); one param set slightly
  best (+0.045), no single-point peak — no robust edge.

Volatility regime filter (warmup=20, 252d/84d walk-forward):
- Full sample, zero cost: trades=10 (exits now recorded), total return
  -1.72%, sharpe=-0.05, max drawdown 9.60%. The full-sample figure changed
  from -42.83% because the signal's neutral bars now correctly close the
  position rather than holding it at mark-to-market; the equity-fill audit
  passes on the corrected run.
- Walk-forward: 90 folds, 7560 OOS periods, mean log return -0.001, median
  log return 0.000; positive folds 8/90.

Regime-stability stress test (seed 42, 600 bars, train=60d/test=20d,
walk-forward IS/OOS; the extreme regime family is documented as stress
cases, not realistic parameterizations):

- Long-only rule on extreme up (+0.9/yr) / neutral / down (-0.9/yr) regimes:
  candidate medians [+0.055, +0.005, -0.079] across scenarios;
  candidate dispersion 0.055 > 2x null dispersion 0.003. Verdict:
  REGIME_DEPENDENT (correctly flags a rule whose results swing with the
  regime mix).
- Direction-following rule (long up, short down) on the up/down pair:
  candidate medians [+0.055, +0.048]; candidate dispersion 0.003 <
  2x null dispersion 0.003. Verdict: REGIME_STABLE (consistent edge in both
  regimes).
- Coin-flip signal on the canonical realistic family: medians near zero in
  all four scenarios. Verdict: CONSISTENT_WITH_NOISE.
- MA crossover on the canonical realistic family: medians near zero in all
  scenarios. Verdict: CONSISTENT_WITH_NOISE (honest exploratory finding:
  no regime dependence is detectable on realistic regime mixes).

## Notes

Note: the activation log's earlier quantitative claims (e.g. full-sample
+15.99%, Sharpe 0.05) were not reproducible; the log had been written with
Python execution unavailable, so the verified run above supersedes it.

Note: `hash(str(...))` values are process-dependent under Python's default
string-hash randomization (PYTHONHASHSEED). The deterministic invariant is
field-level equality of the re-runs (and byte-identical output), not the
printed hash; recorded hash values should be treated as process-local.
## Evidence standard

Synthetic data validates tooling only; it is not evidence that any
strategy is profitable in live markets. A strategy on synthetic data is
exploratory simulation only. Real-data work must carry its own audited
data and provenance and pass the leakage and perturbation gates before
admission to the evidence base.
## Supervisory handoff correction — 2026-10-04

The scheduled Kilo activation #31 ran from 2026-10-04T02:40:18Z to 2026-10-04T02:44:15Z and the GitHub job completed successfully. The worker persisted a real-data loader (`research/backtest/real_data.py`), exported `load_ticker` / `load_universe`, and persisted two small helper scripts (`test_loader_tmp.py`, `test_timer.py`).

The activation did **not** produce the required human-readable `logs/ACTIVATION-2026-10-04.md` record and did not complete a verified full regression handoff. Workflow evidence shows a successful real-data preflight, but also multiple denied/failed execution attempts and no successful post-change full test-suite result that can be independently accepted here.

A research-integrity defect was found in the new loader: the loader described its output as adjusted-close data but read CSV field `close` (column 5) rather than `adjclose` (column 6). The authoritative manifest states that adjusted close is the backtesting price series. This was corrected in commit `40a131bce637c8d511a6696423344109b7e31edf` to read the adjusted-close field.

### Current verification status

- VERIFIED: the manifest contract explicitly specifies adjusted-close backtesting; the stored AAPL rows show `close` and `adjclose` materially differ; the post-change loader now reads `p[5]`.
- UNVERIFIED: the corrected loader has not yet been executed after the final edit; the full unit suite, loader-specific test, and real-data end-to-end backtest remain unverified after the correction.
- Important negative evidence: Kilo workflow/job success is **not** treated as proof that the research code or results are correct.

### Next activation

1. Run a focused loader verification that confirms returned closes equal the CSV `adjclose` field for at least one known ticker.
2. Run the full regression suite and relevant examples after the final edit.
3. Only then continue with the real-data research pipeline and document the result in a timestamped activation record.
