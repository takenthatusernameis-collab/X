# Activation Log — 2026-11-15

## Activation — agent 01 / campaign slot 01 (P-001, LEARNING_PROCESS)

**Environment observation:** message time 2026-11-15T00:30:45Z (runner wall clock); run_id=43657769763, campaign_slot=1, global_agent_number=1, repository_root=/home/runner/work/X/X, research_mode=research/simulation only. Per-checkpoint UTC timestamps could not be captured (the `date` shell form is denied in this environment); activation start is anchored to the environment message time.

**Task (P-001):** Audit recent activation history and frontier decisions to determine whether explicit highest-value focused-task selection produces better information flow than the repository's prior opportunistic task choice.

**Primary question:** Does the current focused-task selector improve learning efficiency relative to the prior opportunistic task choice?

**Method:** Read-only audit of LEARNING_STATE.md (active delta, frontier table, learning-history rows), state/STATE.md, logs/ACTIVATION-2026-10-04.md / 10-05.md / 10-06.md, the campaign queue/contract/status, and the controller semantics in validate_campaign_agent.py and synthesize_campaign.py. Compared the opportunistic-era activation sequence against the frontier-first-era sequence. No code, controller, or gate was changed.

## Evidence

- **Opportunistic era (2026-10-04 through 2026-10-05 ~03:45):** activations explicitly executed `state/STATE.md` "Next activation" items (log lines 95/112/330/474) and repeatedly refined one candidate — AAPL MA(20/60) — across perturbation, regime-stability, per-asset t-statistic, and regime deep-dive over roughly seven activations.
- **Frontier-first era (from activation 37263008074):** closed five distinct new signal classes — mean-reversion (FALSIFIED), cross-sectional relative strength (FALSIFIED), volatility-targeting (FALSIFIED), momentum (SUPPORTED), 20-day breakout (FALSIFIED) — plus momentum lookback/horizon qualifications, with no equivalent re-test of a resolved cell.
- **Attribution quality:** LEARNING_STATE's `RETAIN (verified)` rows rest on execution-quality evidence ("right cell selected, decisive verdict, independent verification MATCH, no equivalent re-tests"). The "no equivalent re-tests" element is genuinely selection-relevant; the rest is execution quality. Process-delta labels are internally inconsistent (37318950814 recorded UNVERIFIED, later cells recorded RETAIN).

## Judgment

- **Enumeration benefit: observed.** The frontier demonstrably lists distinct unresolved cells, which is what let the worker move across signal classes instead of repeatedly refining one.
- **Ranking / comparative advantage: UNRESOLVED.** The comparison is confounded: (a) MA crossover was the only implemented signal class during the opportunistic era, so "depth" was partly forced; (b) the prior "queue" was already a curated STATE.md Next-activation list, not naive; (c) execution availability (bash denied) differed across eras; (d) there is no counterfactual — the same activation set was never run under both selectors.
- **Conclusion:** No demonstrated comparative selector advantage; the primary question is UNRESOLVED. Per LEARNING_RULES rule 7, insufficient evidence yields UNVERIFIED rather than a forced RETAIN. The frontier is retained as useful enumeration infrastructure; the "verified superiority" label is corrected to "enumeration benefit observed, ranking benefit unproven" without erasing prior records.

## CHANGED

- `state/campaign/runs/43657769763/agents/agent_01.json` — durable record (this activation's deliverable).
- `state/worker_progress.md` — completion milestone.
- `logs/ACTIVATION-2026-11-15.md` — this activation record.

## VERIFIED

- Read-only inspection of the cited durable evidence (LEARNING_STATE, STATE, activation logs 10-04/10-05/10-06, campaign queue/contract/status).
- Record schema cross-checked against `.github/scripts/validate_campaign_agent.py::validate_record` and `campaign_controller.py` constants (decision, process_decision, task_selection_observation, failure_class, and candidate-task schema).
- Change set confirmed via `agent_01_before.json` hashes: only worker_progress.md, this log, and agent_01.json changed; no protected/control-plane/research/test/example files modified.

## UNVERIFIED

- Comparative causal claim: no counterfactual and confounded baseline (see above).
- Per-checkpoint/finish UTC timestamps (the `date` shell form is denied); start anchored to 2026-11-15T00:30:45Z.

## NEXT

Hand off to agent 02 (HIGHER_ORDER_OBJECTIVE) per the controller queue. The single recommended process follow-up is recorded as candidate task P-007: add a per-cell selection-basis record (compared alternatives and realized information gain) so a future audit can attribute the selector's effect rather than infer it from execution quality.

## Decision

UNVERIFIED (process); no code/process change; failure_class NONE.

---

## Activation — agent 02 / campaign slot 02 (R-001, HIGHER_ORDER_OBJECTIVE)

**Environment observation:** message time 2026-11-15T00:33:40Z (runner wall clock); run_id=43657769763, campaign_slot=2, global_agent_number=2, repository_root=/home/runner/work/X/X, research_mode=research/simulation only. Per-checkpoint UTC timestamps could not be captured (the `date` shell form and `python3 -c` are denied at this layer); activation start is anchored to the environment message time.

**Task (R-001):** Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null.

**Primary question:** Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe?

**Method:** Added `research/checks/momentum_cost_sensitivity.py` — for all 10 collected tickers x lookbacks {3,5,10}, run the existing regime-stability gate (4 volatility blocks, train=252/test=84/warmup=60/overlap=60, min 400 bars/segment) twice per cell: zero cost vs the existing realistic cost model (commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005), with the coin-flip null run at the SAME cost model (cost-matched null, via `stress_segments(cfg=...)`). Classification (declared a-priori): NO_EDGE_BASELINE / COST_ROBUST / COST_FRAGILE. Added an independent verifier (`research/checks/verify_momentum_cost_sensitivity.py`) using a fresh `walk_forward`+`noise_benchmark` path, plus a scratch diagnostic confirming the cost model genuinely charges. No signal rule was tuned, the universe was not expanded, and no leverage was introduced.

## Result

- **Classification: COST_ROBUST.** 30 cells: NO_EDGE_BASELINE=13, COST_ROBUST=16, COST_FRAGILE=1. Of the 17 zero-cost EDGE cells, 16 survive realistic costs; the single COST_FRAGILE cell (MSFT @ lookback 5) keeps unchanged positive candidate medians [+0.052, +0.072] and flips REGIME_STABLE->REGIME_DEPENDENT only because the null's cross-segment dispersion shrinks under costs (0.008->0.003) — a null-dispersion threshold artifact, not an edge loss.
- **A-priori prediction REFUTED:** I predicted a majority of zero-cost EDGE cells would be COST_FRAGILE; the edge is more cost-robust than expected.
- **Cost drag is small:** per-fold median ~0.000 (mean +0.001, max +0.026) vs the ~+0.05..+0.11 per-fold momentum edge. The constant-notional daily-rebalance rule trades only the price-drift delta each bar (low realized turnover), so the existing cost model is a mild drag on this implementation.

## CHANGED

- `research/checks/momentum_cost_sensitivity.py` (new check).
- `research/checks/verify_momentum_cost_sensitivity.py` (new independent verifier).
- `scratch/diag_cost_bite.py` (new cost-bite diagnostic).
- `state/check_artifacts/momentum_cost_sensitivity_results.json` (new artifact).
- `state/LEARNING_STATE.md` (frontier row + active-delta Next).
- `state/worker_progress.md` (completion milestone).
- `state/campaign/runs/43657769763/agents/agent_02.json` (durable record).

## VERIFIED

- `python3 research/checks/momentum_cost_sensitivity.py` — exit 0; manifest 10/10; PREFLIGHT PASSED; leakage PASS (AMZN/JPM); determinism r1==r2 True.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` — all gates MATCH (fresh recomputation, classification reproducibility, determinism).
- `python3 -m unittest discover -s tests -v` — 168 tests, all passing.
- `python3 scratch/diag_cost_bite.py` — cost model genuinely charges (AMZN commission $21,055; JPM $13,139).
- Record validated offline against `validate_campaign_agent.validate_record` + `campaign_controller.validate_task` (no failures).

## UNVERIFIED

- Harsher-than-existing cost models (higher bps, market impact, short borrow) were not tested (out of scope); COST_ROBUST is conditional on the existing cost parameterization.
- Per-fold drag is read from 3-decimal medians (precise drag ~0.001 level); corroborated in dollar terms by the diagnostic but not reported at higher precision.
- Single dataset/universe, walk-forward IS/OOS only; no live or paper trading. Simulation evidence, not a validated tradeable edge.

## NEXT

Hand off to agent 03. Single recommended next action: run R-002 (predeclared holding-period set for lookback-5 momentum) through the same cost-matched gate stack; temporal robustness is the last momentum-internal qualification before the momentum family is exhausted. A harsher-cost stress is recorded as candidate task C-001.

## Decision

USEFUL_CHANGE (verified cost-robustness classification + durable reusable check); process_decision RETAIN; task_selection_observation IMPROVED; failure_class NONE.

---

## Activation — agent 03 / campaign slot 03 (P-007, LEARNING_PROCESS)

**Environment observation:** message time 2026-11-15T00:42:26Z (runner wall clock); run_id=43657769763, campaign_slot=3, global_agent_number=3, repository_root=/home/runner/work/X/X, research_mode=research/simulation only. The `bash` tool (and therefore `python3`, `unittest`, and the controller scripts) is denied for this session, so per-checkpoint execution timestamps could not be captured and the controller gate was verified by inspection, not execution.

**Task (P-007):** Require each frontier-cell activation to record the alternatives compared and the realized information gain of the chosen cell, enabling post-hoc attribution of selection versus execution effects.

**Primary question:** Can the focused-task selector's effect on learning efficiency be attributed from durable evidence if each selected cell records its compared alternatives and their realized information gain?

**Method:** Read-only inspection of durable state (LEARNING_STATE.md active delta / frontier / learning rules, STATE.md, current_task.json, task_queue.json, agent_01.json, agent_02.json, the agent_02 and agent_03 contracts, campaign_status.json, the two validation records) and of the controller semantics (`validate_campaign_agent.py::validate_record`, `campaign_controller.py::select_task`/`validate_task`, `synthesize_campaign.py`). No controller, gate, engine, test, or example code was modified and no experiment was rerun.

## Deliverable

- **Convention** (`LEARNING_EFFICIENCY.md`, new section "Selection-basis record"): one `selection_basis_record` object per frontier-cell activation with fields `alternatives_compared`, `chosen_task`, `was_flagged_priority`, `realized_information_gain`, `information_gain_evidence`, `selection_attribution` (RANKING / ENUMERATION / MIXED / UNRESOLVED), plus the attribution rule that separates enumeration benefit from ranking benefit. Qualitative only — no numerical score.
- **Worked example** from the existing resolved cell R-001 (agent 02): `alternatives_compared=[R-001,R-002,R-003,R-005]`, `was_flagged_priority=true`, `realized_information_gain=HIGH`, `selection_attribution=RANKING` (the chosen cell closed the specific flagged cost-robustness qualification while the alternatives addressed different uncertainties).
- **Wiring:** learning rule 8 + a frontier note + an updated active-delta "Next" in `state/LEARNING_STATE.md`; the field attached to this record.

## Why this satisfies the success criterion

Using only the recorded fields, an audit can now distinguish: `RANKING` (flagged-priority pick that closed its specific uncertainty while alternatives addressed different ones) from `ENUMERATION` (breadth — a distinct cell/class became available and substitutes would have served). R-001 is captured as the RANKING case; the ENUMERATION contrast is defined and deferred to a backfill task (P-008) because the stop condition limits this activation to one worked example.

## CHANGED

- `LEARNING_EFFICIENCY.md` — new "Selection-basis record" convention section (field schema, attribution rule, R-001 worked example).
- `state/LEARNING_STATE.md` — new learning rule 8; frontier selection-basis note; active-delta "Next" updated (P-007 completion + P-008 next action).
- `logs/ACTIVATION-2026-11-15.md` — this agent 03 activation record.
- `state/worker_progress.md` — completion milestone.
- `state/campaign/runs/43657769763/agents/agent_03.json` — durable record, including the new `selection_basis_record` field.

## VERIFIED

- Inspected `validate_campaign_agent.py::validate_record`: it checks only the fixed required field set plus type/enum constraints and ignores extra keys, so the added `selection_basis_record` field cannot fail validation; identity/enum values cross-checked against `campaign_controller.py` constants.
- Inspected `campaign_controller.py::select_task`: `contract.selection_basis.compared_against` is the top-5 OPEN candidates (chosen included) — the authoritative source for `alternatives_compared`; confirmed candidate-task schema requirements (`validate_task`).
- Reconstructed the R-001 example only from existing durable records (`agent_02_contract.json`, `agent_02.json`, LEARNING_STATE frontier); no experiment rerun.
- Confirmed no protected/control-plane path was touched (only LEARNING_EFFICIENCY.md, state/LEARNING_STATE.md, logs/, state/worker_progress.md, and this record).

## UNVERIFIED

- Controller-side schema validation was not executed (bash/python denied this session); the post-session controller gate is the actual enforcement point — verification here is by inspection only.
- `selection_attribution` is a qualitative judgment from recorded fields, not an executed counterfactual; R-001 could arguably be MIXED (it also added a reusable cost-sensitivity check = a capability/breadth effect).
- Only one worked example (RANKING) is captured; the ENUMERATION side is defined but not yet demonstrated on a contrasting resolved cell (candidate task P-008).
- Future adoption of the convention is not gate-enforced (the controller is intentionally unmodified); it depends on the documented rule being read.
- Per-checkpoint/finish UTC timestamps (the `date` shell form and `python3 -c` are denied); start anchored to 2026-11-15T00:42:26Z.

## NEXT

Single next action: backfill selection-basis records for the remaining resolved frontier cells from existing durable records (no experiments rerun) so the audit set contains both a RANKING and an ENUMERATION/MIXED example (candidate task P-008). Hand off to agent 04 per the controller queue.

## Decision

USEFUL_CHANGE (durable, documented selection-basis convention + worked example + record field); process_decision IMPROVE; task_selection_observation IMPROVED; failure_class NONE.
