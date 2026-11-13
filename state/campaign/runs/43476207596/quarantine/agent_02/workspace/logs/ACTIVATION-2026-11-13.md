# Activation Log — 2026-11-13

## Activation — Global Agent 01 (campaign slot 01, task P-001, LEARNING_PROCESS)

**Observed environment:** message time 2026-11-13T17:56:45Z (runner wall clock); run_id 43476207596, campaign slot 01, global agent 01, ref main. Precise finish/checkpoint timestamps were not observable (the `date` shell form is denied in this environment); the start is anchored to the observed message time and no timestamps are backfilled.

**Task (P-001):** Audit recent activation history and frontier decisions to determine whether explicit highest-value focused-task selection (frontier-first) produces better information flow than the prior opportunistic, queue-driven task choice. Deliverable: one bounded, evidence-backed process judgment.

**Selection basis:** frontier-first (controller `task_rank` qualitative ordering; highest evidence-backed qualitative priority among OPEN LEARNING_PROCESS tasks).

## Work performed (read-only audit)

1. Read the selector mechanism: `campaign_controller.py::task_rank` / `candidate_tasks_for_role` (ordinal qualitative ordering of the hand-authored `task_queue`, no numeric score) and the LEARNING_STATE frontier-first protocol.
2. Read durable evidence: `state/LEARNING_STATE.md`, `state/STATE.md`, `logs/ACTIVATION-2026-10-05.md`, `logs/ACTIVATION-2026-10-06.md`, `state/campaign/task_queue.json`, `current_task.json`, `campaign_status.json`, `evaluate_research_process.py`, `synthesize_campaign.py`, `validate_campaign_agent.py`.
3. Independently confirmed the frontier cells are distinct signal classes (dedicated check + independent verifier + artifact each: mean_reversion, cross_sectional_relative_strength, volatility_targeting, momentum, momentum_lookback_sweep, momentum_longer_horizon, breakout_20day_cont) and that the opportunistic baseline is a single activation (37260572520, whose objective "came from a queued receipt item rather than a frontier cell" and whose process delta is UNVERIFIED).

## Judgment

- **Descriptive effect (observed):** frontier-first selection produced a coherent, non-redundant, independently-verified sequence of decisive frontier closures with no equivalent duplicate re-test, and yielded the first genuine positive candidate (momentum SUPPORTED, 7/10 REGIME_STABLE). This is a real improvement in information flow over the single documented opportunistic activation.
- **Causal advantage (NOT demonstrated):** the opportunistic baseline is n=1 (37260572520); the frontier-first period is confounded by full framework maturity and by the specific (positive) momentum cell; the controller's `task_rank` orders a hand-authored queue, not the LEARNING_STATE frontier, so no controlled counterfactual exists.
- **Decision:** RETAIN the frontier-first selection protocol (no observed bottleneck; it demonstrably works). No process change, no numeric reward score, no controller redesign (out of scope).

## CHANGED

- `state/campaign/runs/43476207596/agents/agent_01.json` (new: the required process-decision record).
- `state/worker_progress.md` (semantic checkpoint: audit complete, decision recorded).
- `logs/ACTIVATION-2026-11-13.md` (this activation record).

No research/code/process artifact was modified.

## VERIFIED

- `python3 -m json.tool state/campaign/runs/43476207596/agents/agent_01.json` — valid JSON; all 23 required fields present; `decision=RETAIN`, `process_decision=RETAIN`, `task_selection_observation=IMPROVED`, `failure_class=NONE`; `changed`/`verified`/`unverified`/`candidate_tasks` are arrays; `next` is a single bounded line.
- Evidence inspections listed in the record's `verified` array (selector mechanism, distinct frontier cells, no equivalent re-tests, n=1 opportunistic baseline, frontier verdicts cross-checked against `momentum_lookback_sweep_results.json`).

## UNVERIFIED

- Causal superiority of the selector over opportunistic choice is not established (n=1 baseline; framework-maturity and cell-choice confounds; queue-vs-frontier ordering mismatch).
- The no-equivalent-retest / novelty-guard property is not independently stress-tested (separate open task P-002).
- No per-checkpoint wall-clock timestamps beyond the observed message time.

## NEXT

Task P-002: audit resolved frontier cells and task history for duplicate/near-duplicate work and determine whether a minimal novelty guard is sufficient (the operative mechanism behind the observed frontier-first information-flow benefit).

---

## Activation — Global Agent 02 (campaign slot 02, task R-001, HIGHER_ORDER_OBJECTIVE)

**Observed environment:** message time 2026-11-13T18:00:12Z (runner wall clock); run_id 43476207596, campaign slot 02, global agent 02, ref main. The `date` shell form is denied; the start is anchored to the observed message time and no timestamps are backfilled. Finish/checkpoint timestamps were not observable.

**Task (R-001):** Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null; classify cost robustness. Deliverable: a reproducible artifact and independent verification result.

**Selection basis:** frontier-first; highest evidence-backed qualitative priority among OPEN HIGHER_ORDER_OBJECTIVE tasks (the momentum cost-qualification cell).

## Work performed

1. Inspected durable state (`state/LEARNING_STATE.md`, `state/STATE.md`, `current_task.json`), the momentum check family (`momentum.py`, `momentum_lookback_sweep.py`, `momentum_longer_horizon.py`), the cost model (`examples/ma_crossover_real_data.py`: `commission_per_trade=2.0`, `commission_per_share=0.003`, `slippage_cents=2.0`, `slippage_proportional=0.0005`), and the engine/`stress_segments`/`noise_benchmark` cost plumbing.
2. **Discovered a verified implementation defect** in `research/backtest/regime_stability.py::momentum_signals` and `::mean_reversion_signals`: both compute `mean(log(price))` over the lookback window (always positive for adjusted close > 1) and take its sign, so momentum is a constant LONG (4460/4460 non-neutral bars +1, 0 short) and reversal a constant SHORT, independent of lookback. Confirmed empirically (AAPL/AMZN/JPM/XOM) and via a corrected return-based signal (~800-900 flips / ~4.5k bars). The prior "momentum SUPPORTED / reversal FALSIFIED / lookback-robust / longer-horizon" results are therefore constant long/short (beta/drift) and the lookback robustness is vacuous.
3. Wrote and ran `research/checks/momentum_cost_sensitivity.py`: cost grid {zero, realistic, 2x, 4x} x the existing realistic cost model, lookbacks 3/5/10, matched coin-flip null, for (A) the admitted candidate as implemented and (B) a defect-repair diagnostic (previous lookback-day return). Wrote artifact `state/check_artifacts/momentum_cost_sensitivity_results.json`.
4. Wrote and ran `research/checks/verify_momentum_cost_sensitivity.py` (independent path: fresh `walk_forward` + `noise_benchmark`, no `stress_segments`; cross-consistency vs `momentum_lookback_sweep_results.json`; classification reproducibility; determinism). All four gates MATCH.

## Result

- **Existing (constant-long) candidate:** COST_ROBUST = 7 (AAPL, MSFT, GOOGL, AMZN, META, TSLA, JPM); NO_EDGE = 3 (NVDA, JNJ, XOM). Identical at lookbacks 3/5/10 (degeneracy signature). Medians ~+0.05..+0.13 essentially unchanged from zero to 4x cost; low turnover (3.81x annualized on AMZN; $21,055 realistic commission). Cost-robust **because it is beta, not because momentum survives costs**.
- **Corrected return-based momentum diagnostic:** NO_EDGE = 10/10 at every lookback (no robust positive edge; ~105x turnover; $284,746 realistic commission on AMZN; cost-fragile where marginally positive).
- **Primary question answered:** the short-horizon momentum edge does not survive cost stress because there is no robust short-horizon momentum edge as implemented or corrected; the prior positive evidence is an artifact of the signal defect.

## CHANGED

- `research/checks/momentum_cost_sensitivity.py` (new).
- `research/checks/verify_momentum_cost_sensitivity.py` (new).
- `state/check_artifacts/momentum_cost_sensitivity_results.json` (new).
- `state/STATE.md` (appended durable CORRECTION section).
- `state/LEARNING_STATE.md` (prepended CORRECTION block; superseded the prior Next).
- `state/worker_progress.md` (semantic checkpoint).
- `state/campaign/runs/43476207596/agents/agent_02.json` (required record).
- `logs/ACTIVATION-2026-11-13.md` (this entry).

No framework, test, or existing-check file was modified.

## VERIFIED

- `python3 research/data/preflight.py` — PREFLIGHT PASSED (all 10 tickers).
- `python3 research/checks/momentum_cost_sensitivity.py` — ran end-to-end; determinism r1 == r2; artifact written.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` — Gate 1 (fresh walk_forward + noise_benchmark, AMZN/JPM x both variants x lookbacks 3/5/10 x all cost levels) MATCH; Gate 2 (existing zero-cost lookback-5 vs `momentum_lookback_sweep_results.json`, 10 tickers) MATCH; Gate 3 (classifications reproduce) MATCH; Gate 4 (determinism) identical.
- `python3 -m unittest discover -s tests -q` — 168 tests OK.
- Empirical defect confirmation: constant long (all +1) / constant short (all -1) vs corrected signal flipping ~800-900 times.

## UNVERIFIED

- No per-checkpoint wall-clock timestamps beyond the observed message time.
- The corrected return-based momentum signal exists only as a diagnostic in the check; it is not yet in the framework, unit-tested, or used to re-run the momentum/reversal cells (candidate task R-002).
- A correctly implemented momentum/reversal signal might still show an edge at some horizon/cost; this activation only establishes the shipped implementation is degenerate and the corrected daily-rebalance signal has no robust positive edge here.

## NEXT

Candidate task R-002: repair `momentum_signals`/`mean_reversion_signals` to use the previous lookback-day return, add unit tests, and re-run the momentum (with cost grid) and mean-reversion cells through the gate stack with a fresh independent verifier.