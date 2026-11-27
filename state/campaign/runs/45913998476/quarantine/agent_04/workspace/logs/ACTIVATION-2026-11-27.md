# Activation 2026-11-27 — Global Agent 2 (campaign slot 02, run 45913998476)

- **Activation start (observed):** 2026-11-27T00:00:59Z (controller message time; the
  `date`/`stat` shell forms are denied at this layer, so no independent finish clock
  was captured — see UNVERIFIED).
- **Role:** HIGHER_ORDER_OBJECTIVE
- **Task:** R-001 — bounded cost-sensitivity check for lookback 3/5/10 momentum under
  the repository's existing realistic cost model and matched null.
- **Primary question:** Does the short-horizon momentum edge survive conservative
  transaction-cost stress on the collected real-data universe?
- **Mode:** research / simulation only. No live trading or production execution.

## CHANGED

- `research/checks/momentum_cost_sensitivity.py` (new): the bounded cost-sensitivity
  check. Predeclared cost grid {zero, realistic, stress2x} × lookback {3,5,10} across
  the fixed 10-asset universe via the authoritative `stress_segments_across_tickers`
  path; the matched coin-flip null is run under the SAME cfg (costs) as the candidate
  inside `stress_segments`. Realistic = the repo's existing model copied verbatim from
  `examples/ma_crossover_real_data.py` (commission_per_trade=2.0,
  commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005);
  stress2x doubles every component. Artifact precision 6 dp.
- `research/checks/verify_momentum_cost_sensitivity.py` (new): independent verifier.
  Gate 1 fresh `walk_forward` + `noise_benchmark` recomputation (AMZN/JPM, all lb ×
  cost); Gate 2 classification reproducibility from artifact gaps; Gate 3 zero-cost
  lookback-5 cross-consistency vs `momentum_results.json`; Gate 4 determinism.
- `state/check_artifacts/momentum_cost_sensitivity_results.json` (new): the artifact.
- `state/campaign/runs/45913998476/agents/agent_02.json` (new): durable record.
- `state/worker_progress.md`: milestone update.
- `research/checks/_diag_cost_tmp.py`: throwaway diagnostic used to confirm the cost
  cfg reaches the candidate path (turnover/commission measurement). `rm` is denied at
  this layer, so it is left in place; it is NOT part of the evidence base.

No framework, controller, workflow, dependency, or universe changes.

## VERIFIED (exact commands that succeeded after the final edits)

- `python3 research/checks/momentum_cost_sensitivity.py` — manifest 10/10 OK; PREFLIGHT
  PASSED; leakage PASS (AMZN/JPM); equity-fill audit PASS under zero AND realistic
  costs; determinism r1==r2 True; artifact written. OVERALL (canonical lookback 5):
  **COST_ROBUST**; lookbacks classified COST_ROBUST = 3/3.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` — Gate 1 MATCH (AMZN/JPM
  × lookback 3/5/10 × zero/realistic/stress2x, candidate medians and matched-null
  medians); Gate 2 classification MATCH; Gate 3 all 10 tickers zero-cost lookback-5
  MATCH vs `momentum_results.json`; Gate 4 determinism True.
- `python3 -m unittest discover -s tests -v` — **168 tests, all OK** (unchanged; the new
  checks are standalone and not imported by the suite).

## Result (durable)

Cost-robustness classification (predeclared rule: an asset is in the edge set when its
per-segment candidate-minus-null gap is strictly positive in every segment;
COST_ROBUST iff the zero-cost edge set is preserved at realistic cost):

| lookback | class | \|E0\| | \|Er\| | \|E2\| |
|---|---|---|---|---|
| 3 | COST_ROBUST | 8 | 8 | 7 |
| 5 (canonical) | COST_ROBUST | 9 | 9 | 8 |
| 10 | COST_ROBUST | 8 | 8 | 8 |

E0 at lookback 5 = {AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, TSLA, XOM}; NVDA is the
only excluded asset (a stable loser: first-segment median -0.038 → -0.064 → -0.088 as
costs rise, gap negative — correctly not an edge).

Mechanism (important, honest): the candidate's absolute net-of-cost medians are
essentially unchanged by costs (per-fold drag ~1e-4); the candidate-minus-null gap
actually WIDENS under costs because the matched coin-flip null churns more (random daily
weights) and pays more costs than the persistent momentum signal. So COST_ROBUST is
driven by (a) the candidate's own low absolute cost drag and (b) the null paying more.

A-priori prediction (recorded before execution): COST_MARGINAL or COST_FRAGILE (costs
expected to materially erode a daily-rebalanced rule). **The prediction was REFUTED** —
the edge is cost-robust under the repo's cost model and a 2× stress.

Verdict-label note: MSFT flips REGIME_STABLE→REGIME_DEPENDENT and JPM flips
REGIME_DEPENDENT→REGIME_STABLE between zero and realistic cost. These are null-dispersion
artifacts (the candidate medians are flat); the gap-based classification is the robust
read, consistent with the documented "verdict labels are not self-interpreting" caveat.

## UNVERIFIED / caveats

- Finish timestamp not captured (`date`/`stat` denied); only the observed start time is
  recorded. No timestamp was invented.
- "Cost-robust" is conditional on the engine's constant-dollar (fixed-notional) sizing:
  traded notional per bar is small relative to the 1e6 base and cumulative P&L, so the
  absolute cost drag is ~1e-4/fold. A compounding / equity-scaled sizing, or materially
  higher slippage, could produce larger costs — NOT tested here (bounded grid; out of
  scope). This is the main residual qualification.
- Only one realistic cost point + a 2× stress were tested; the broader cost surface is
  not mapped (bounded by design).
- The gap-widening is partly a matched-null turnover effect, not purely a candidate
  property.
- JNJ and XOM are in E0 but are noise-level (CONSISTENT_WITH_NOISE) with small positive
  gaps; the E0 composition is reported so COST_ROBUST is not over-read.

## NEXT (one bounded action)

Execute R-002: test a fixed lookback-5 momentum signal with a predeclared small
holding-period set through the existing walk-forward/null methodology (temporal
robustness — the next decision-relevant qualification now that cost robustness is
resolved).

---

# Activation 2026-11-27 — Global Agent 3 (campaign slot 03, run 45913998476)

- **Activation start (observed):** 2026-11-27T00:10:11Z (controller message time; the
  `date`/`stat` shell forms are denied, so no independent finish clock was captured).
- **Role:** LEARNING_PROCESS
- **Task:** P-002 — audit resolved frontier cells and task history for duplicate or
  near-duplicate work; determine whether a minimal novelty guard is sufficient.
- **Primary question:** Does the durable history adequately prevent repeating
  equivalent experiments when a frontier cell is already resolved?
- **Mode:** research / simulation only. No live trading or production execution.

## Decision: RETAIN (process)

Resolved-cell duplication is prevented; the residual gap is at the implementation-
artifact layer and is recorded, not realized.

## Evidence (durable)

- Resolved cells (MA crossover, mean reversion, CSRS, volatility targeting, momentum,
  cross-sectional momentum, 20-day breakout) are all marked FALSIFIED/TESTED/REJECTED
  with "None - cell closed" in `state/LEARNING_STATE.md`; RESOLVED tasks (R-001, R-004)
  are excluded from OPEN in `state/campaign/task_queue.json`; no resolved cell was
  re-executed. The momentum follow-ons (lookback sweep, longer horizon, cost) are
  distinct information objectives the guard explicitly permits.
- Residual gap: `research/checks/momentum_sweep.py` (lookbacks 3/5/10/20) and
  `research/checks/momentum_hold_period_sensitivity.py` (R-002 holding-period check)
  exist with NO artifact, NO verifier, and NO activation record. `momentum_sweep.py`
  is an unexecuted near-duplicate of the executed `momentum_lookback_sweep.py`
  (3/5/10) + `momentum_longer_horizon.py` (20/60). `momentum_hold_period_sensitivity.py`
  is the fully written but unexecuted R-002 implementation while R-002 is still OPEN,
  so a fresh R-002 agent could re-derive it.
- No realized duplicate execution was found; the risk is forward-looking.

## CHANGED

- `state/campaign/runs/45913998476/agents/agent_03.json` (new: P-002 record).
- `state/worker_progress.md`: milestone update.
- `logs/ACTIVATION-2026-11-27.md`: appended this section.

## VERIFIED (inspection; bash denied)

- Frontier table closures + selector RESOLVED exclusion + no re-runs (LEARNING_STATE,
  task_queue, campaign_status, agent_01.json).
- `research/checks/` vs `state/check_artifacts/` listings: two orphan scripts with no
  artifact/verifier; grep confirms no activation record references them.

## UNVERIFIED

- No command execution (bash denied); findings rest on durable-record inspection.
- Orphan-script provenance not recorded; status inferred from missing artifacts/records.
- Finish timestamp not captured; no timestamp invented.

## NEXT (one bounded action)

Future LEARNING_PROCESS activation: add one sentence to the LEARNING_EFFICIENCY section 2
novelty guard requiring a `research/checks/` glob and recording of superseded/orphaned
scripts before writing a new check (candidate task P-008), then controller-validate.