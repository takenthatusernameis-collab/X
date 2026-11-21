# Activation Record — 2026-11-21

## Activation — campaign 44900285443, slot 01 / global agent 1 (task P-001, LEARNING_PROCESS)

**Observed activation window:** start anchored to the environment message time
**2026-11-21T11:42:20Z**. Finish time **UNVERIFIED** — the `date` shell form is
denied in this environment (`{"permission":"bash","pattern":"*","action":"deny"}`),
so no wall-clock could be read. Run identity: run_id 44900285443, campaign_slot 1,
global_agent_number 1, event_sha 618539a4a9627acf46f68bc72b855293fd64c871, ref main.

**Objective (task P-001):** Audit recent activation history and frontier decisions to
determine whether explicit highest-value focused-task selection produces better
information flow than the repository's prior opportunistic task choice.

**Primary question:** Does the current focused-task selector improve learning
efficiency relative to the repository's prior opportunistic task choice?

**Selection basis (controller-recorded):** frontier-first / qualitative
evidence-backed priority; `compared_against` = [P-001, P-002, P-004, P-003, P-006].

## What was inspected (durable evidence)

- `state/LEARNING_STATE.md` — active strategy delta, frontier table, learning history.
- `state/STATE.md` — section index and the `Next activation` lists that drove the
  opportunistic era.
- `logs/ACTIVATION-2026-10-02.md` … `logs/ACTIVATION-2026-10-06.md` — all five
  human-readable activation records.
- `state/campaign/task_queue.json`, `runs/44900285443/task_queue_initial.json`,
  `campaign_status.json`, `current_task.json`, `state/activation_status.json`.
- `.github/scripts/campaign_controller.py` (`task_rank`, `candidate_tasks_for_role`,
  `select_task`, `validate_task`) and `.github/scripts/validate_campaign_agent.py`
  (read only; protected control-plane files were not modified).

## Findings

1. **Selector mechanism verified (positive).** `task_rank`
   (`campaign_controller.py:206-218`) is a 7-key lexicographic ordinal sort
   (expected_information_gain, downstream_leverage, falsification_power,
   verifiability, then effort, complexity, execution_risk, then `task_id`). No
   numeric score. Role and OPEN status are filtered first.
2. **Ranking recomputed from durable queue values.** P-001 is the *unique* minimum at
   (HIGH, HIGH, HIGH, HIGH, LOW, LOW, LOW); P-002 / P-004 rank next
   (falsification_power MEDIUM); P-003 / P-006 tie at (MEDIUM, HIGH, HIGH, HIGH, LOW,
   LOW, LOW) and are ordered by `task_id`; P-005 ranks last. The recorded
   `compared_against` list equals `candidates[:5]` from this ordering, so contract,
   code and queue are consistent. The only tie (positions 4-5) is immaterial to the
   selected task; no tie-break degeneracy was observed at the top.
3. **Opportunistic era (evidence of the prior failure mode).** Three executed
   activations on 2026-10-05 (00:13 per-asset t-statistic, 02:21 regime-dependence
   deep-dive, 03:45 regime-adaptive MA = 37260572520) took objectives from the linear
   STATE.md `Next activation` list and all stayed inside the single MA-crossover
   class. LEARNING_STATE lines 51-52 record 37260572520 and 37261348530 as UNVERIFIED
   process deltas (queued-receipt objective; execution blocked).
4. **Frontier-first era (evidence of broader coverage).** Seven logged activations
   (37263008074, 37304873966, 37314710995, 37318950814, 37361000967, 37394344201,
   37414038601) each closed a distinct frontier cell with a decisive verdict and an
   independent-verification MATCH: five signal-class cells (reversal, cross-sectional
   relative strength, volatility targeting, momentum, 20-day breakout) plus two
   momentum qualifications. An eighth frontier-first activation (cross-sectional
   momentum, activation id not recorded in logs or `STATE.md`) closed a sixth signal
   class per `LEARNING_STATE.md` line 34 and its artifact
   (`cross_sectional_momentum_results.json`). No equivalent re-test is recorded.
5. **Counter-evidence preserved.** Three of the seven frontier-first activations
   (37318950814, 37361000967, 37394344201) stayed inside the momentum family — the
   "refine the most interesting recent result" pattern that
   `LEARNING_EFFICIENCY.md` section 1 warns against.
6. **Confound preserved.** 37260572520's own NEXT action declared MA-crossover
   exhausted and prescribed a new signal class, so class exhaustion — not selection
   alone — explains part of the shift to new cells. The comparison is observational,
   not controlled.
7. **Record-completeness gap.** Activation 37394344201 and the cross-sectional-
   momentum activation (artifact `cross_sectional_momentum_results.json`, lookback
   63, concentration NO_EDGE) have LEARNING_STATE rows but no `STATE.md` section and
   no log entry.
8. **Durable-state inconsistency.** LEARNING_STATE line 12 records the frontier-first
   delta as `RETAIN (verified)`; line 57 records activation 37318950814's
   per-activation process decision as `UNVERIFIED (process delta)`.

## CHANGED

- `state/campaign/runs/44900285443/agents/agent_01.json` — new durable agent record.
- `logs/ACTIVATION-2026-11-21.md` — this activation record.
- `state/worker_progress.md` — phase and verified-milestone checkpoint.

No controller, workflow, queue, or research file was modified. No mechanism was
redesigned and no numeric reward score was introduced.

## VERIFIED (post-change inspection)

- The agent record parses as a single JSON object and contains every field required by
  `validate_campaign_agent.validate_record`, with `agent_number` 1, `campaign_slot` 1,
  `global_agent_number` 1, `role` LEARNING_PROCESS, `task_id` P-001, and canonical
  enum values (`decision` VERIFIED_NEGATIVE_RESULT, `process_decision` RETAIN,
  `task_selection_observation` UNCHANGED, `failure_class` NONE).
- The single candidate task P-007 carries every field in
  `campaign_controller.REQUIRED_TASK_FIELDS`, a single-question
  `primary_question`, valid severities, and a boolean `tests_preceding_process`.
- `next` is exactly one non-empty line.

## UNVERIFIED

- Finish timestamp (no clock access; `date` denied).
- Controller-side validation of this record was not executed by the agent (bash
  denied); the workflow's `validate_campaign_agent.py` gate remains the verifying step.
- The regression suite was not run because no research, test, or example code was
  touched.
- The causal effect of selection on information flow (no isolating comparison exists).

## Decision

`VERIFIED_NEGATIVE_RESULT` — the selector's efficiency advantage is **not
demonstrated** by durable evidence, while the selector mechanism itself is verified
deterministic, role-aware, contract-bound and uniquely top-ranked. Process decision
`RETAIN`: no evidence of harm, verified structural guarantees, redesign out of scope.

## NEXT

In campaign slot 03, write one pre-declared non-numeric measurement rule (decisive
frontier cells closed per activation and equivalent re-tests per activation, computed
from durable agent records) so a later synthesis can test the selector's efficiency
claim without a reward score.