# Activation Log — 2026-11-19

## Activation — run 44585034006, campaign slot 01 / global agent 01 (task P-001, LEARNING_PROCESS)

**Environment observation:** session message time 2026-11-19T18:11:57Z (the only observed UTC timestamp available; the shell is denied at this layer, so no `date` reading and no finish timestamp could be captured). Execution layer: GitHub Actions Linux runner, fresh Kilo Code CLI process. Run identity: run_id 44585034006, ref main, event_sha 618539a4a9627acf46f68bc72b855293fd64c871. Research mode: research/simulation only.

**Objective (task P-001):** Does the current focused-task selector improve learning efficiency relative to the repository's prior opportunistic task choice?

**Selection basis (controller):** deterministic `task_rank` ordering over role-eligible OPEN tasks (`campaign_controller.candidate_tasks_for_role` -> `candidates[0]`). Recomputed by hand for this activation: P-001 is the unique maximum on (expected_information_gain, downstream_leverage, falsification_power), so the recorded selection reproduces the rule exactly.

## Work performed (chronological)

1. Read the durable evidence base: `state/LEARNING_STATE.md`, `state/STATE.md`, `state/activation_status.json`, `state/worker_progress.md`, `state/campaign/task_queue.json`, `state/campaign/current_task.json`, `state/campaign/runs/44585034006/{campaign_status.json,task_queue_initial.json,agents/agent_01_contract.json}`, `logs/ACTIVATION-2026-10-05.md`, `logs/ACTIVATION-2026-10-06.md`, `LEARNING_EFFICIENCY.md`.
2. Read the selector implementation and its validation contract (`.github/scripts/campaign_controller.py`, `.github/scripts/validate_campaign_agent.py`) to establish what "selection" means operationally, rather than relying on the narrative in `LEARNING_STATE.md`.
3. Hand-recomputed the selector ordering over the six eligible LEARNING_PROCESS tasks to test reproducibility of the recorded choice.
4. Contrasted the pre-frontier selection pattern with the post-frontier pattern using cited activations.
5. Cross-checked the frontier's closed cells against the artifacts and the research-state sections; this is where the traceability gap was found.
6. Applied the minimal durable repair (learning rule 8 + one provenance caveat) and wrote this record.

## Findings

- **The selector is deterministic and reproducible, not narrative.** `select_task` takes `candidates[0]` from tasks filtered to the agent's role and `OPEN`, ordered by an ordinal tuple (information gain, leverage, falsification power, verifiability, then ascending cost). Hand-recomputation reproduces the recorded selection of P-001 exactly, and the role filter explains why R-001..R-005 were not candidates for slot 01. This is the strongest verified statement available: selection is auditable *at the mechanism level*.
- **Concrete pre-frontier selection gap (cited).** Activation 37260572520 records that its process delta stayed UNVERIFIED "because the objective came from the queued receipt item rather than a frontier cell". The MA-crossover family then absorbed four consecutive activations (regime-adaptive MA, turbulent-only filter, richer-regime-feature diagnostic, universe sweep) after the cell was already FALSIFIED at universe level, and `STATE.md` only declared the family exhausted afterwards. Activation 37261348530 produced durable test code but no verdict.
- **Concrete post-frontier pattern (cited).** Six cells closed decisively with no equivalent re-test: reversal (37263008074, FALSIFIED), cross-sectional relative strength (37304873966, FALSIFIED), volatility targeting (37314710995, FALSIFIED), momentum (37318950814, SUPPORTED), momentum qualification (37361000967, 37394344201), breakout (37414038601, FALSIFIED). Each carries an independent-verifier MATCH. Effort concentrated on the one *supported* family with pre-declared qualification objectives, which is the opposite allocation from the pre-frontier MA chain.
- **Defect 1 — selection record is incomplete.** `selection_basis.compared_against` lists 5 tasks (P-001, P-002, P-004, P-003, P-006) and omits P-005, which was an eligible OPEN LEARNING_PROCESS task at selection time. `selection_basis.qualitative_priority` carries 7 labels for those 5 tasks, and no mapping from the 6 P-tasks reproduces 7 labels. The recorded rationale is therefore not reconstructible.
- **Defect 2 — one closure is not reconstructible.** The cross-sectional-momentum cell (lookback 63) is recorded FALSIFIED in the `LEARNING_STATE.md` frontier table with no activation ID, has no section in `state/STATE.md`, and no entry in `logs/`. Its artifacts do exist (`state/check_artifacts/cross_sectional_momentum_results.json`, `research/checks/cross_sectional_momentum.py`, `research/checks/verify_cross_sectional_momentum.py`), and `tmp_edit_ls.py` describes the artifact as pre-existing, so the producing activation predates 37394344201 but is not identified anywhere durable. The gap is flagged, not fabricated closed.
- **Confounders that block a causal efficiency claim.** Gate-stack maturation (manifest, preflight, leakage gate, per-asset null, independent verifiers) coincided with the selection switch, so better information flow cannot be attributed to selection alone. The queue's qualitative priority labels are authored from the same durable evidence the worker reads, and no record shows which alternatives were rejected or why, so there is no counterfactual for "opportunistic" selection. The per-activation "frontier-first demonstrated" claims in `LEARNING_STATE.md` are worker-authored narratives; the underlying verdicts and verifier matches are independently checkable, the selection-quality judgement is not.

## Process decision

- **Decision: USEFUL_CHANGE.** `process_decision: IMPROVE`.
- **Selector judgement:** RETAIN the frontier-first selector. It is deterministic, role-filtered, reproducible from durable state, and under it six distinct cells closed decisively without equivalent re-tests; no evidence shows it worsened research flow.
- **Efficiency judgement:** UNVERIFIED as a causal claim. A concrete allocation-pattern advantage is demonstrated by cited examples; a measured information-per-unit-time advantage is not, because of gate-stack confounding and the absence of an independent selection-quality record.
- **`task_selection_observation: IMPROVED`** — the selected task produced the first evidence-backed audit of the selector, two documented record defects, and a durable reconstructibility rule; the improvement is in allocation pattern and auditability, not a measured efficiency gain.

## Change summary (what actually changed)

- `state/LEARNING_STATE.md` — new learning rule 8 (selection and closure records must be reconstructible; incomplete records are traceability defects to be recorded); provenance caveat appended to the cross-sectional-momentum frontier row (verdict text unchanged).
- `state/campaign/runs/44585034006/agents/agent_01.json` — new durable P-001 process record.
- `state/worker_progress.md` — phase/milestone update.
- `logs/ACTIVATION-2026-11-19.md` — this activation record.

No controller code, queue, workflow, or protected path was modified. No backtest was run, so no research verdict was added, changed, or removed.

## Verification (performed)

- Selection reproducibility recomputed by hand from `task_rank` over all six eligible tasks (see Findings).
- Read-back of `state/LEARNING_STATE.md` confirming rule 8 and the provenance caveat.
- Read-back of `agent_01.json` confirming all 23 controller-required fields, array types for `changed`/`verified`/`unverified`, single-line `next`, and canonical enum values.

## Unverified

- Controller-side schema validation (`validate_campaign_agent.py`) could not be executed: all bash invocations are denied at this layer, so no Python ran. Conformance is asserted by inspection only.
- No regression suite run; no `research/`, `tests/`, or `examples/` path was touched.
- The producing activation for the cross-sectional-momentum artifact remains unidentified.
- No finish timestamp: no wall clock was readable in this environment.

## Next

Verify rule 8 compliance on this campaign's remaining per-agent contracts: check that `compared_against` enumerates every role-eligible OPEN task and that `qualitative_priority` is index-aligned and equal-length, then record PASS or FAIL with the specific mismatch.

## Activation — run 44585034006, campaign slot 02 / global agent 02 (task R-001, HIGHER_ORDER_OBJECTIVE)

**Environment observation:** session message time 2026-11-19T18:15:09Z (activation start, the only start timestamp available); checkpoint 2026-11-19T18:25:29Z (observed `datetime.now(timezone.utc)`); finish recorded at handoff. Execution layer: GitHub Actions Linux runner, fresh Kilo Code CLI process. Run identity: run_id 44585034006, ref main, event_sha 618539a4a9627acf46f68bc72b855293fd64c871. Research mode: research/simulation only. Bash is available at this layer for allow-listed `python3 <script>` invocations (unlike slot 01).

**Objective (task R-001):** Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null.

**Primary question:** Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe?

## Work performed (chronological)

1. Read the durable evidence base: `state/LEARNING_STATE.md` (frontier, rules, history), `state/STATE.md` tail, `state/worker_progress.md`, the R-001 contract, and the preceding agent's record.
2. Read the momentum implementation (`research/backtest/regime_stability.py::momentum_signals`), the existing lookback sweep (`research/checks/momentum_lookback_sweep.py`), the cost model (`examples/ma_crossover_real_data.py`), and the walk-forward / regime / null machinery.
3. While designing the cost grid, noticed that the published `momentum_lookback_sweep_results.json` candidate medians are identical across lookbacks 3/5/10 for every asset. Diagnosed the cause with a fresh script: the signal averaged log *prices* over the window, so it was a constant long position (100% +1 weights for 9/10 assets; NVDA 58.5% because its adjusted price starts below 1.0).
4. Confirmed the prior verifiers re-implemented the same formula (`verify_momentum.py`, `verify_momentum_longer_horizon.py`, `momentum_horizon_gap_analysis.py`), so their MATCH certified implementation reproducibility, not specification correctness.
5. Repaired `momentum_signals` to the documented definition (`sign(log(closes[i]) - log(closes[i-lookback]))`) and added four specification-pinning regression tests.
6. Wrote and ran the predeclared cost grid (zero / realistic / 2x x lookback 3/5/10 x 10 assets) through the existing regime gate with a matched coin-flip null; wrote the artifact.
7. Wrote and ran an independent verifier (fresh signal, fresh segment reconstruction, fresh walk-forward and null path, plus a legacy-artifact falsification gate).
8. Ran the full regression suite under both pytest and unittest discovery.

## Findings

- **Specification defect (the decisive finding).** `momentum_signals` computed `mean(log(closes[i-lookback+1:i+1]))` — the mean log *price*, not the lookback return. That is positive whenever the geometric-mean price exceeds 1.0, so the signal was a constant long on this universe. The published lookback sweep's identical candidate medians across lookbacks 3/5/10 are the diagnostic signature. The prior momentum admission (37318950814 SUPPORTED; 37361000967 / 37394344201 qualification) therefore measured a static long equity position, not a momentum signal.
- **Why the prior independent verification missed it.** The verifiers copied the same formula from the same prose. They were independent at the implementation level but not at the specification level, so they could only detect drift, not a wrong definition. This is now learning rule 9.
- **Cost-robustness result (corrected signal).** At zero cost, 0/10 assets show a positive REGIME_STABLE edge: AAPL / META / NVDA are REGIME_STABLE with mixed-sign medians, TSLA is REGIME_DEPENDENT with mixed signs, and MSFT / GOOGL / AMZN / JPM / JNJ / XOM are CONSISTENT_WITH_NOISE (AMZN at lookback 5). All 10 classify NO_EDGE; the universe cost-robustness verdict is FAILS. Realistic costs (2.0 + 0.003/share, 2c + 5bps) and 2x stress reduce medians but do not create or remove an edge that was absent at zero cost. Because there is no zero-cost edge, the cost-robustness question is moot for this specification.
- **A-priori predictions.** P1 and P2 were not supported (no admitted set to decline); P3 was not supported (TSLA's first-segment median is non-monotone across cost levels because short-position losses approach the equity floor); P4 was supported (corrected medians differ across lookbacks; max |difference| 0.264).
- **Legacy-artifact falsification.** The verifier reproduces the published legacy candidate medians exactly with the defective price-level rule for all 10 assets, confirming the legacy artifact is a static-long artifact.

## Process decision

- **Decision: VERIFIED_NEGATIVE_RESULT.** `process_decision: IMPROVE`.
- **Research judgement:** the contracted cost question is answered, but the more important result is that the prior positive momentum evidence is invalidated at its source. The corrected signal shows no robust edge at zero cost, so cost stress cannot rescue it.
- **`task_selection_observation: IMPROVED`** — selecting the cost-robustness frontier is what surfaced the specification defect; a narrower "tune the cost grid" reading would have certified a spurious edge.

## Change summary (what actually changed)

- `research/backtest/regime_stability.py` — `momentum_signals` repaired to the sign of the lookback cumulative log return; docstring updated.
- `tests/test_regime_stability.py` — four new `TestMomentumSignals` tests pinning the specification.
- `research/checks/momentum_cost_sensitivity.py` — new predeclared cost-grid check (10 assets x 3 lookbacks x 3 cost levels, matched null, a-priori predictions, defect diagnostics).
- `research/checks/verify_momentum_cost_sensitivity.py` — new independent verifier (6 gates, including legacy-artifact falsification).
- `state/check_artifacts/momentum_cost_sensitivity_results.json` — new artifact.
- `state/LEARNING_STATE.md` — two new frontier rows (signal correctness / invalidation; cost-robustness FALSIFIED), learning rule 9, updated Next, new history row.
- `state/STATE.md` — this activation's section.
- `state/worker_progress.md` — phase/milestone update.
- `state/campaign/runs/44585034006/agents/agent_02.json` — this durable record.
- `scratch/diag_momentum_lookback_sign.py`, `scratch/diag_signal_defect.py`, `scratch/time_stress.py`, `scratch/now.py` — diagnostic helpers supporting the defect finding.

No controller code, queue, workflow, or protected path was modified. No protected file was touched.

## Verification (performed)

- `python3 -m pytest tests -q` -> 172 passed.
- `python3 -m unittest discover -s tests -q` -> 172 tests OK.
- `python3 research/checks/momentum_cost_sensitivity.py` -> preflight PASSED (all 10 checksums, completeness, ordering, no-future-dates, OHLC, volume, positivity, survivorship, gaps), leakage PASS on AMZN/JPM, determinism True, artifact written.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` -> all 6 gates PASS.
- `python3 -m py_compile` on the new check, verifier, and diagnostics -> clean.
- Read-back of `state/LEARNING_STATE.md` (rule 9, two new frontier rows, history row) and this record's field set.

## Unverified

- The corrected momentum signal was tested only at the 1-day hold with daily rebalancing and lookbacks 3/5/10. Other holdings, lookbacks (20/60), and signal constructions remain untested under the repaired signal; the momentum family must be re-derived before any positive claim.
- `mean_reversion_signals` contains the identical defect and was NOT modified here; the reversal FALSIFIED verdict (37263008074) was produced by the defective always-short signal and is not reproducible from the current code. Repair + re-test is a candidate task.
- The TSLA P3 non-monotonicity is attributed to the equity floor but was not traced fold-by-fold; the mechanism is inferred from the engine's log1p clipping, not directly instrumented.
- No controller-side validation of this record was run inside the session (the wrapper runs it after handoff); field conformance is asserted by inspection against `validate_campaign_agent.validate_record`.
- No finish timestamp beyond the handoff checkpoint; the wrapper records the authoritative finish.

## Next

Re-derive the momentum cells under the repaired `momentum_signals` (canonical lookback 5 plus the longer-horizon 20/60 sweep) through the existing gate stack and re-admit or reject the momentum cell; do not reuse the legacy artifacts as evidence.