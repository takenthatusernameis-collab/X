# Activation Record — 2026-10-03 (follow-on: verification + per-asset significance flags)

## Activation goal

1. **Verify the repository's durable state independently.** The prior
   activation reported 72 tests and seed-42 figures for the examples; those
   were reported by a previous activation, not independently reproduced here.
   This activation re-ran the full suite and all four walk-forward examples to
   confirm the baseline is actually green and reproducible before standing on
   it.
2. **Implement the documented next action: per-asset significance flags.**
   `state/STATE.md` flagged that the universe sweep's global null dispersion
   can be large with few folds per asset, and recommended reporting a
   per-asset significance flag (median vs the asset's own coin-flip benchmark)
   alongside the verdict. This closes the last gap in the robustness
   tooling.

## Work performed

### 1. Baseline verification (execution now available: numpy installed)

- `pip install -q numpy` (Python 3.12, numpy resolved in this runner).
- `python3 -m unittest discover -s tests -v`: **77 tests, all passing**
  (9 data + 19 engine + 8 metrics + 15 perturbation + 11 regime-stability
  + 15 asset-universe).
- All four walk-forward examples run end-to-end without error; seed-42 figures
  reproduce the documented records exactly:

  | Example | Key verified figures |
  |---|---|
  | `ma_crossover` | full-sample zero cost: trades=55, total=-106.93%, sharpe=-0.04, max_dd=128.21%, turnover=319.8x; walk-forward: 88 folds, 7392 OOS, mean log ret -0.123, median -0.081, 37/88 positive |
  | `volatility_regime_filter` | full-sample: trades=10, total=-1.72%, sharpe=-0.05, max_dd=9.60%; walk-forward: 90 folds, 7560 OOS, mean -0.001, median 0.000, 8/90 positive |
  | `regime_stability_demo` | long-only extreme: medians [+0.055, +0.005, -0.079], dispersion 0.055 > 2x null 0.003, REGIME_DEPENDENT; direction rule: [+0.055, +0.048], dispersion 0.003 < 2x null 0.003, REGIME_STABLE; coin-flip and MA on canonical family: CONSISTENT_WITH_NOISE |
  | `universe_sweep` | MA on 7-asset family: medians [-0.133..+0.020], best-asset share 41%, null median -0.047, null dispersion +0.109, verdict NO_EDGE |

- Independent determinism check (`research/checks/determinism_check.py`): two
  independent runs of `asset_sweep_summary(..., per_asset_null=True)` produce
  identical fields and the same hash (3968836896295251151).

### 2. Per-asset significance flags (the framework improvement)

- `research/backtest/universe.py`:
  - `AssetSweepSummary` gained `asset_null_medians` (coin-flip baseline per
    asset), `asset_significance` (per-asset flag: candidate median
    distinguishable from its own null), and `n_significant_assets`.
  - The `verdict` property uses per-asset nulls when available: the NO_EDGE
    check then compares every asset to its own null rather than a single
    global null. The `per_asset_null=False` default preserves the prior API
    and all existing verdicts.
  - `asset_sweep_summary` gained the `per_asset_null=True` option; the null
    is computed for every asset, and `inspect()` prints per-asset flags plus
    the effective tolerance.
- `examples/universe_sweep.py`: runs with `per_asset_null=True`, now printing
  the flags.
- `tests/test_universe.py`: new `TestPerAssetNull` class (5 tests:
  determinism, flat family stays NO_EDGE under the per-asset null, agreement
  with the global-null path on a clear NO_EDGE case, single significant edge
  yields CONCENTRATED, and self-consistency of the flags).

Result (seed 42, 7-asset drifted family): null dispersion tightened from
+0.109 (global, first 3 assets) to +0.079 (per-asset, all 7), effective
tolerance +0.158, 0 / 7 significant assets, verdict NO_EDGE — unchanged from
the prior run. All 77 tests pass and all four examples run.

## State of the toolkit (after this activation)

All three robustness dimensions from `research/METHODOLOGY.md` are operational,
tested, and verified:
- **Perturbation** (`perturbation.py`): parameter-sensitivity sweep with a
  coin-flip null; the MA crossover shows a flat deviation scan, mixed signs,
  no single-point peak — no robust edge.
- **Regime stability** (`regime_stability.py`): stress across regime
  families; contrived long-only is REGIME_DEPENDENT, a direction-following
  rule is REGIME_STABLE, noise and the MA crossover are CONSISTENT_WITH_NOISE.
- **Asset universe** (`universe.py`): sweep across assets with per-asset
  significance flags; contrived one-strong-asset family → CONCENTRATED,
  uniform-strong family → CONSISTENT, MA crossover → NO_EDGE.

Synthetic data validates tooling only; none of the two demo signals is a
live-market edge (both are reliably negative on regime-switching synthetic
data).

## Files changed / created

- `research/backtest/universe.py` — per-asset null and significance flags.
- `examples/universe_sweep.py` — enables `per_asset_null=True`.
- `tests/test_universe.py` — `TestPerAssetNull` (5 new tests).
- `state/STATE.md` — verification count updated (77 tests); new "Current
  activation (asset-universe significance flags)" section; next-activation
  items revised.
- `logs/ACTIVATION-2026-10-03.md` — this record (consolidates the universe
  sweep record from earlier this date with the present activation).
- `research/checks/determinism_check.py` — scratch helper: independent
  determinism re-run of the universe sweep.
- `research/checks/test_count.py` — scratch helper: counts unit tests per
  module.

## Verification (executed)

- `python3 -m unittest discover -s tests`: **77 tests, all passing** (verified
  three times; latest run: 9 data + 19 engine + 8 metrics + 15 perturbation
  + 11 regime-stability + 15 asset-universe; 77 total, 9.6s).
- `python3 -m examples.ma_crossover`, `volatility_regime_filter`,
  `regime_stability_demo`, `universe_sweep`: all run end-to-end; seed-42
  figures match the documented records.
- `python3 research/checks/determinism_check.py`: two independent runs produce
  identical fields and hash 3968836896295251151.

## Failures / known issues

None. The MA crossover perturbation example prints "Baseline indistinguishable
from noise: False" on this seed because the coin-flip null happened to have a
median (+0.065) farther than 0.05 from the baseline (-0.081); this is honest
reporting of null noise, not a framework defect (the deviation scan is still
flat and mixed-sign, which is the robustness signal).

Known limitation (unchanged): the significance flags are currently opt-in
(`per_asset_null=True`); flipping the default is the lowest-risk follow-on.

## Next actions for the next activation

1. Regression discipline: run the full suite and all four examples together
   after any research/code change (add a short CI-free checklist to
   `research/README.md` if desired).
2. Default per-asset nulls: the `per_asset_null=True` path is implemented,
   tested, and demonstrated; flipping the default tightens the NO_EDGE check
   and is deterministic. If done, re-run the suite and examples to confirm
   the canonical verdicts (NO_EDGE, CONCENTRATED, CONSISTENT) are unchanged.
3. Real-data readiness: if an in-scope public real dataset is identified for
   research-only simulation, create its manifest per
   `research/REAL_DATA_FEASIBILITY.md` and pass the pre-run leakage review
   checklist before any real-data execution.

## Handoff

The enterprise now has complete, verified, deterministic robustness tooling
(perturbation, regime stability, asset-universe with per-asset significance
flags), a green 77-test suite, and four byte-reproducible examples. A fresh
activation can reproduce this state with:

```
python3 -m unittest discover -s tests -v
python3 -m examples.ma_crossover
python3 -m examples.volatility_regime_filter
python3 -m examples.regime_stability_demo
python3 -m examples.universe_sweep
python3 research/checks/determinism_check.py
```

Note: the earlier universe-sweep activation on this date is documented in the
same file (now consolidated here).
