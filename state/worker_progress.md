# Worker Progress Checkpoint

## Current objective
Test whether a **regime-adaptive MA crossover** (fast windows in turbulent
regimes, standard windows in calm) on the REGIME_DEPENDENT assets (AMZN, JPM)
produces REGIME_STABLE results, or remains REGIME_DEPENDENT / CONSISTENT_WITH_NOISE —
falsifying that the 2009-2013 edge is separable by a past-only volatility-regime
classifier.

## Current phase
COMPLETE — independent evidence gate ran and passed in this activation.

## Last VERIFIED milestone (this activation, 2026-10-05T03:13Z onward)
- **Smoke test**: `python3 research/checks/regime_adaptive_ma.py` exits 0;
  manifest 10/10 checksums OK; PREFLIGHT PASSED (all data-quality/survivorship
  checks incl. known-gaps audit); leakage PASS on AMZN and JPM; regime-adaptive
  MA run on both assets plus the 10-asset universe generalization printed.
- **Independent verification**: `python3 research/checks/verify_regime_adaptive.py`
  exits 0; all six published figures (AMZN/JPM base, adaptive, turbulent_only)
  recomputed via a fresh `walk_forward` implementation (not `stress_segments`)
  MATCH exactly; all six deterministic across reruns.
- **Regression**: `python3 -m unittest discover -s tests -v`: 120 tests, all
  passing.
- **Defects found and fixed this activation**: `research/checks/regime_adaptive_ma.py`
  had a `sys.path` off-by-one (`parent.parent.parent.parent` -> `parent.parent.parent`);
  `research/checks/verify_regime_adaptive.py` docstring stated wrong figures
  (universe-wide segment medians) that conflicted with its own assertions; both
  repaired and re-run green.

## Evidence / artifact produced
Falsification of the hypothesis "a regime-adaptive MA yields a REGIME_STABLE edge
on the REGIME_DEPENDENT assets":

  AMZN: base    [0.046, 0.001, -0.154] REGIME_DEPENDENT   (dispersion +0.086)
        adaptive [-0.028, 0.001, 0.015] CONSISTENT_WITH_NOISE (dispersion +0.018)
        turbulent_only [0.046, 0.0, -0.154] REGIME_DEPENDENT (dispersion +0.086)
  JPM:  base    [0.109, -0.006, 0.033] REGIME_DEPENDENT   (dispersion +0.047)
        adaptive [0.069, -0.006, -0.001] REGIME_DEPENDENT (dispersion +0.034)
        turbulent_only [0.109, 0.0, 0.033] REGIME_DEPENDENT (dispersion +0.045)

Universe-wide (adaptive variant on all 10 collected tickers):
CONSISTENT_WITH_NOISE=5, REGIME_STABLE=4, REGIME_DEPENDENT=1 (JPM only).
Both the `stress_segments` path and the fresh `walk_forward` recomputation path
agree on these figures.

Interpretation: shortening MA windows in turbulent regimes removed the AMZN
swing only by eliminating the edge (CONSISTENT_WITH_NOISE); on JPM the
dependence persists even with adaptive windows (dispersion ~9x the coin-flip
null on the filtered variant). A past-only volatility classifier cannot separate
the good 2009-2013 turbulent regime from the bad 2022-2026 turbulent regime, so
the 2009-2013 edge is not harvestable as a regime-contingent strategy. Negative
conclusion: the regime-adaptive MA is not admitted to the evidence base.

## File changes this activation (working tree, not yet persisted)
- `research/backtest/regime_stability.py` — new
  `regime_adaptive_ma_signals(...)`, plus signature-aware `labels` routing in
  `stress_segments` (signal functions that classify on the fly route by the same
  labels that defined the segments).
- `research/backtest/__init__.py` — export `regime_adaptive_ma_signals`.
- `research/checks/regime_adaptive_ma.py` — new; sys.path fix applied.
- `research/checks/verify_regime_adaptive.py` — new; docstring figures repaired.

## Long-running?
No. All execution completed (universe example ~1-2s; full adaptive check on 2
assets ~20-25s; independent verification ~40s; suite 120 tests in ~15s).

## Next bounded action
Fold the regime-adaptive MA verdicts into `state/STATE.md` as the next durable
research conclusion (negative: the regime-adaptive MA does not rescue a
REGIME_STABLE edge; not admitted to the evidence base).
