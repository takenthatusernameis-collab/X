# Worker Progress

activation_id: 37304873966
phase: SMOKE / DEEP
status: IN_PROGRESS

## objective
Execute and close the cross-sectional relative strength (CSRS) frontier cell
(long winners / short losers, rank-based sizing signal class on the collected
10-asset adjusted-close universe) through the perturbation + coin-flip-null +
regime-stability gates, with an independent verifier. Record the verdict in
state/STATE.md, state/LEARNING_STATE.md, state/activation_status.json, and the
day log. Result: FALSIFIED — CSRS is not a robust edge on the collected
universe; segment medians are negative and within the coin-flip null band,
concentration NO_EDGE, synthetic sweep CONSISTENT_WITH_NOISE; engine-path
cross-check MATCH on every fold, determinism r1==r2 across two full runs.

## last verified milestone
CSRS helpers added to research/backtest/regime_stability.py (csrs_spread_daily_returns,
csrs_null_spread_daily_returns, csrs_null_spread_family, build_synthetic_spread_asset,
csrs_spread_family) + __init__.py exports; smoke test hand-verified against a tiny
synthetic family; four defects repaired (off-by-one loops, missing export, engine-path
warmup/compounding mismatch -> per-fold constant-share signals, sweep keying +
determinism-assert bugs); check ran end-to-end twice: manifest 10/10 OK, preflight
passed, leakage review, engine self-consistency PASS, engine cross-check MATCH on
both segments, NO_EDGE concentration, sweep CONSISTENT_WITH_NOISE, determinism r1==r2;
artifact written to state/check_artifacts/cross_sectional_relative_strength_results.json.

## next bounded action
Update state/activation_status.json and logs/ACTIVATION-2026-10-05.md with the CSRS
verdict, then run the 120-test regression suite to re-establish the regression
baseline after the helper edits.

## long-running expected?
No — the CSRS check completed (~4 min, ~96 synthetic walk-forwards on 800-bar
families plus ~84 per-segment folds); the regression suite is the next step.

---
This file is a liveness contract. Update it only after a real research-state
transition or verified milestone.
