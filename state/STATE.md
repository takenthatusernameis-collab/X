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
re-ran with 77 tests passing, and `research/checks/determinism_check.py`
reports `r1 == r2: True`. See `logs/ACTIVATION-2026-10-03.md`.

## Next activation

1. Regression discipline: run the full test suite and all four walk-forward
    examples together after any research/code change to catch regressions
    early.
2. Real-data readiness: if an in-scope public real dataset is identified for
    research-only simulation, create its manifest per
    `research/REAL_DATA_FEASIBILITY.md` and pass the pre-run leakage review
    checklist before any real-data execution.

## Verification (executed)

The framework is deterministic and the full suite executes:

- `pip install -q numpy` (resolved numpy 2.5.3); all modules compile.
- `python -m unittest discover -s tests -v`: **77 tests, all passing**
  (9 data + 19 engine + 8 metrics + 15 perturbation + 11 regime-stability
  + 15 asset-universe); 0 failures/errors, including the new assertion that
  the `asset_sweep_summary` default path reports per-asset significance
  flags.
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
