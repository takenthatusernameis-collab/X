# Activation 2026-11-22 — Campaign slot 02 (Global Agent 2)

- task_id: R-001 (HIGHER_ORDER_OBJECTIVE)
- objective: Run a bounded cost-sensitivity check for lookback 3/5/10 momentum
  under the repository's existing realistic cost model and matched null.
- activation_start_utc: 2026-11-22T18:42:23Z (controller before-capture;
  harness message time 2026-11-22T18:42:27Z)
- finish_utc: UNVERIFIED (no independent clock available in this session; `date`
  is denied by project policy — not backfilled)
- environment: GitHub Actions Linux runner, fresh Kilo Code CLI, research/simulation only

## Questions answered before substantive work
1. Becoming better at: classifying whether the only admitted positive signal
   (short-horizon momentum) survives execution-cost stress before admission.
2. Binding constraint: all momentum evidence is zero-cost; cost robustness of a
   mechanical daily-rebalanced 1-day bet was unmeasured.
3. Uncertainty blocking a better decision: does the positive gross edge remain
   net-positive after realistic costs at lookbacks 3/5/10?
4. Single bounded action: predeclared 4-level cost grid x 3 lookbacks x 10-asset
   universe through the existing regime/null framework, then independent recompute.
5. Worth it: HIGH priority qualification of the only positive candidate, LOW
   execution risk, existing infrastructure, decisive either way.

## CHANGED
- research/checks/momentum_cost_sensitivity.py (new): predeclared cost grid
  (m in {0.0,0.5,1.0,2.0} x realistic model commission_per_trade=2.0,
  commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005)
  for lookback {3,5,10} across the 10-asset collected universe, matched coin-flip
  null at the same cost config, a-priori cost classes, zero-cost cross-consistency
  vs the prior sweep, and an instrument audit.
- research/checks/verify_momentum_cost_sensitivity.py (new): independent verifier
  (fresh walk_forward + inlined null; zero-cost tie to prior sweep; classification
  reproduction; insolvency-audit reproduction; determinism).
- state/check_artifacts/momentum_cost_sensitivity_results.json (new artifact).
- scratch/diag_cost_monotonicity.py, scratch/diag_cost_fold.py (new diagnostics
  that located the cost-accounting defect).

## VERIFIED (exact commands that succeeded after the final edits)
- `python3 research/checks/momentum_cost_sensitivity.py` -> exit 0; manifest 10/10
  OK; preflight PASSED; leakage PASS; 30/30 zero-cost cells MATCH the prior
  `momentum_lookback_sweep_results.json`; determinism identical.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` -> exit 0; all 5
  gates MATCH (AMZN/JPM recompute; 10-ticker zero-cost tie to prior; a-priori
  classes reproduce; NVDA null-insolvency reproduced -168252.9 -> -522727.4,
  total_return +2.107; determinism identical).
- `python3 -m unittest discover -s tests` -> Ran 168 tests, OK.

## Result (predeclared rule, reported for transparency)
- Per-lookback cost class: COST_ROBUST at all three lookbacks (edge-asset counts
  stable at 5-6/10 across m=0.0..2.0). Overall SURVIVES_REALISTIC_COST;
  survives 2x conservative = True.

## Result (decision-relevant, after instrument audit)
- UNVERIFIED. The existing cost model is scale-dependent (per-share
  commission_per_share=0.003 and slippage_cents=2.0). Under the engine's
  constant-notional sizing (target_shares = weight*capital/close), low-priced
  split-adjusted series (NVDA ~$0.2-0.4 in 2009-13) require millions of shares,
  so per-fill costs reach ~$14k commission + ~$95k slippage, driving the matched
  coin-flip NULL into insolvency (NVDA lookback-5 m=0.5 first-segment OOS start
  equity = -168,252.9). The engine metric total_return = eq[-1]/eq[0]-1 then
  returns a meaningless +2.107 from a negative/negative ratio, inflating null
  dispersion and making REGIME_STABLE trivially achievable. The candidate medians
  are stable (e.g. universe mean 0.0613 -> 0.0595), but the instrument cannot
  support a decision-grade cost-robustness claim. Zero-cost momentum evidence is
  intact (zero-cost column reproduces the prior sweep exactly).

## UNVERIFIED
- finish timestamp (no clock).
- Whether a scale-invariant (percentage-of-notional) cost model would change the
  verdict — not tested here (would be a new cost model, out of this task's scope).

## NEXT (one bounded action)
- Repair the cost model to be scale-invariant (percentage-of-notional commission
  and slippage) and make total_return solvent-only (reject/clip negative-equity
  folds), then rerun this same predeclared grid and verifier.