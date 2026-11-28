# Activation Log — 2026-11-28

## Activation — campaign 46092417292, agent 02 (task R-001, momentum cost-sensitivity)

**Environment observation:** message time 2026-11-28T01:20:35Z (activation start); run identity GITHUB_RUN_ID=46092417292, GITHUB_RUN_ATTEMPT=1, event SHA=618539a4a9627acf46f68bc72b855293fd64c871, REF=main, repository root=/home/runner/work/X/X, execution layer = GitHub Actions Linux runner, fresh Kilo Code CLI process. Finish timestamp observed 2026-11-28T01:28:06Z (via `python3 scratch/now.py`; the `date` shell form is denied in this environment). No intermediate per-step wall-clock timestamps were captured; checkpoints below are ordered, not precisely timed.

**Objective (task R-001, HIGHER_ORDER_OBJECTIVE):** Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe? Bounded action: cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null.

**Selection basis:** highest evidence-backed qualitative priority among OPEN tasks for this role (compared against R-001/R-002/R-003/R-005). The preceding agent (P-001) established that the frontier-first selection practice improves information flow; this activation executes the frontier's Next action (the decision-relevant cost qualification of the only admitted positive candidate).

**A-priori design (predeclared, not tuned):** cost grid {zero, realistic, conservative} x lookback {3,5,10} x 10 tickers. Realistic = the repository's canonical model (commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005, from examples/ma_crossover*.py); conservative = 2x variable costs (10 bps proportional). Same walk-forward (train=252/test=84/warmup=60/overlap=60, 4 past-only volatility blocks, min 400 bars/segment) and a cost-matched coin-flip null (stress_segments forwards the same BacktestConfig to candidate and noise_benchmark). Classification predeclared: COST_ROBUST / COST_FRAGILE / NO_EDGE from the framework EDGE rule, plus a null-independent candidate-only view and a max candidate-median-shift diagnostic.

## Work performed (chronological)

1. **Inspected durable evidence:** LEARNING_STATE.md (frontier + history + active delta), STATE.md, worker_progress.md, campaign_status.json, current_task.json, task_queue.json (R-001..R-005), the momentum check + lookback sweep + verifier, engine.py (BacktestConfig cost model), regime_stability.py (stress_segments/momentum_signals), perturbation.py (noise_benchmark/parameter_sweep), and examples/ma_crossover*.py (canonical realistic cost model). Confirmed python3 script execution works (bash `ls`, `python3 <file>`, `python3 -m unittest` succeed; `find`, `python3 -c`, `date`, and compound/redirect commands are denied at this layer).

2. **Wrote the check** `research/checks/momentum_cost_sensitivity.py` (predeclared grid, cost-matched null, verdict + candidate-only classification, turnover diagnostic, determinism, artifact).

3. **Ran the check** — manifest 10/10 OK, preflight PASSED, leakage PASS (AMZN/JPM), determinism r1==r2 True, artifact written.

4. **Diagnosed an anomaly before trusting results:** candidate medians were near-identical across cost levels for high/medium-priced assets while NVDA shifted sharply. `scratch/diag_cost.py` + `scratch/diag_cost2.py` showed the engine DOES apply costs to the candidate (AAPL full-sample commission $15,721 conservative vs $0 zero; final equity differs) and that the near-zero shift is a genuine small-relative-to-edge rebalancing cost under constant-dollar sizing, whereas NVDA (early adjusted price ~$0.20 -> millions of shares) incurs ~$453k commission and erodes (-0.038 -> -0.088). The identical-across-lookbacks candidate medians are a pre-established framework property (activation 37394344201).

5. **Refined the classification** to add a null-independent candidate-only view + max candidate-median shift, so a null-dispersion verdict flip (JPM) is not mistaken for a candidate change; reran the check.

6. **Wrote and ran the independent verifier** `research/checks/verify_momentum_cost_sensitivity.py` — 4 gates, fresh path (walk_forward + noise_benchmark, no stress_segments): Gate 1 AMZN/JPM all lookbacks x costs MATCH; Gate 2 zero-cost lookback-5 vs momentum_lookback_sweep_results.json all 10 MATCH; Gate 3 classification/universe counts reproduce MATCH; Gate 4 determinism identical.

7. **Regression suite:** `python3 -m unittest discover -s tests -p 'test_*.py'` — 168 tests, OK, 0 failures.

8. **Records:** this log, `state/worker_progress.md` milestone, and `state/campaign/runs/46092417292/agents/agent_02.json`.

## Results

- **Candidate-only (null-independent):** 9/10 assets have positive candidate segment medians at both zero and conservative cost; **none eroded by cost** (max candidate-median shift <= 0.002 for 8/10; 0.05 only for NVDA, already NO_EDGE).
- **Verdict-based (framework EDGE = REGIME_STABLE with uniformly positive medians):** COST_ROBUST=7, COST_FRAGILE=0, NO_EDGE=3 (NVDA, JNJ, XOM).
- **Lookback-5 (admitted cell) survives conservative cost:** 6/10 assets (AAPL, GOOGL, AMZN, META, TSLA, JPM). Note: this cost-based lookback-5 survivor set differs from the prior lookback-robust set — under cost, MSFT's lookback-5 verdict becomes REGIME_DEPENDENT (it still survives at lookbacks 3/10, so remains COST_ROBUST overall) while JPM's lookback-5 becomes REGIME_STABLE.
- **Mechanism:** under the repo's constant-dollar sizing the per-bar rebalancing cost is small relative to the edge for medium/high-priced assets; the per-share commission only materially bites the low-priced asset (NVDA). JPM's verdict-based COST_ROBUST is a null-dispersion flip (candidate unchanged), reported honestly via the candidate-only view.

## Verdict

**RETAIN (positive qualification).** The short-horizon momentum edge is NOT a turnover/cost artifact under the repository's existing realistic cost model: the candidate's positive P&L is essentially unchanged by realistic and 2x conservative costs for the lookback-robust assets, and no asset's positive candidate is eroded by cost. This strengthens (does not overturn) the admitted candidate. Independently verified (4 gates MATCH) and deterministic; 168/168 regression tests pass.

## Change summary (what actually changed)

- `research/checks/momentum_cost_sensitivity.py` — new bounded cost-sensitivity check.
- `research/checks/verify_momentum_cost_sensitivity.py` — new independent verifier (4 gates).
- `state/check_artifacts/momentum_cost_sensitivity_results.json` — new artifact.
- `scratch/inspect_cost.py`, `scratch/diag_cost.py`, `scratch/diag_cost2.py`, `scratch/now.py` — diagnostics (retained per PERSISTENCE_POLICY).
- `state/campaign/runs/46092417292/agents/agent_02.json`, `state/worker_progress.md`, `logs/ACTIVATION-2026-11-28.md` — records.
- No change to any signal rule, the universe, leverage, the controller, or any workflow file.

## Verification (exact post-change checks)

- `python3 research/checks/momentum_cost_sensitivity.py` — exit 0; manifest 10/10; preflight PASSED; leakage PASS; determinism r1==r2 True; artifact written.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` — Gate 1/2/3/4 all MATCH/identical.
- `python3 -m unittest discover -s tests -p 'test_*.py'` — Ran 168 tests, OK.
- `python3 scratch/validate_agent02.py` — agent_02.json schema OK (23 required fields, 2 candidate_tasks).

## Unverified

- No single frozen "realistic cost model" constant exists in the repo; the canonical examples model + a 2x stress were used. A pure bps-only (percent-of-notional) cost model and other venue fee schedules were not tested (recorded as candidate task R-007).
- The test is more sensitive to the per-share commission on low-priced assets than to a uniform bps drag under constant-dollar sizing.
- No strict temporal holdout (select on early data, evaluate the most-recent untouched period once) was run (recorded as candidate task R-006 and the single `next` action).
- No per-step UTC timestamps captured (only start 01:20:35Z and finish 01:28:06Z observed).

## Next

Task R-006 (proposed): strict temporal holdout of the lookback-5 momentum edge — select on data through 2023-12-31, evaluate the untouched 2024-01-01..2026-10-03 period once with the realistic cost cfg and matched null, and classify persistence.

---

## Activation — campaign 46092417292, agent 03 (task P-002, duplicate-work / novelty-guard audit)

**Environment observation:** message time 2026-11-28T01:30:01Z (activation start, observed); agent_03_before.json captured_utc 2026-11-28T01:29:56Z; run identity GITHUB_RUN_ID=46092417292, GITHUB_RUN_ATTEMPT=1, event SHA=618539a4a9627acf46f68bc72b855293fd64c871, REF=main, repository root=/home/runner/work/X/X, execution layer = GitHub Actions Linux runner, fresh Kilo Code CLI process. The `bash` tool and the `date` shell form are denied at this layer, so all inspection used repository-native read/grep/glob and no finish wall-clock timestamp was captured (none invented).

**Objective (task P-002, LEARNING_PROCESS):** Audit resolved frontier cells and task history for duplicate or near-duplicate work and determine whether a minimal novelty guard is sufficient. Primary question: does the durable history adequately prevent repeating equivalent experiments when a frontier cell is already resolved? Out of scope: rerunning historical experiments or adding broad metadata systems.

**Selection basis:** highest evidence-backed qualitative priority among OPEN LEARNING_PROCESS tasks (compared against P-004/P-003/P-006/P-005). The preceding agent (R-001) qualified the admitted momentum candidate; this activation audits the learning process itself (duplicate avoidance), not the market.

## Work performed (chronological, read-only)

1. Inspected durable state: LEARNING_STATE.md (frontier table, learning history, active delta, Learning rules), STATE.md, worker_progress.md, task_queue.json (P-001..P-007, R-001..R-007 with statuses), current_task.json, campaign_status.json (selection_history + status gating), agent_01.json (P-001), agent_02.json (R-001).
2. Inventoried research/checks/ (33 files) against state/check_artifacts/ (12 artifacts); classified each momentum-family check by information objective.
3. Repo-wide search for `momentum_sweep` references to detect orphans (found only its own ARTIFACT_PATH line, the agent before-snapshot hashes, and this session's log — no artifact, no verifier, no markdown reference).
4. Confirmed the other checks lacking a JSON artifact are distinct diagnostics (cross_sectional_regime_diagnostic, regime_dependent_deep_dive, regime_adaptive_ma, momentum_hold_period_sensitivity=R-002 pending), not near-duplicates.

## Findings (concrete duplicate-avoidance evidence)

- **Experiment-level prevention works.** Each distinct frontier-first cell from activation 37263008074 onward was closed exactly once with a decisive verdict, independent-verification MATCH, and determinism, with no equivalent re-test: mean-reversion FALSIFIED, cross-sectional relative strength FALSIFIED, volatility targeting FALSIFIED, momentum lookback-5 ADMITTED, momentum lookback-sweep 3/5/10 TESTED, momentum longer-horizon 20/60 TESTED, cross-sectional momentum 63d FALSIFIED, 20-day breakout FALSIFIED.
- **Novelty guard already codified** in LEARNING_STATE Learning rules #2 and LEARNING_EFFICIENCY section 2. The OPEN queue momentum variants (R-002 holding period, R-003 volatility gate, R-005 sizing, R-006 strict holdout, R-007 bps-only cost) each carry a distinct information objective and satisfy the guard's exception criteria.
- **Queue status gating** skips RESOLVED cells (R-001, R-004); selection_history shows no repeated task_id.
- **One concrete code-level near-duplicate:** `research/checks/momentum_sweep.py` is an orphan near-duplicate of the executed `momentum_lookback_sweep.py` (and overlaps `momentum_longer_horizon.py`) — same universe, walk-forward, volatility blocks, coin-flip null, and per-asset regime-stability metric over lookbacks {3,5,10} (momentum_sweep adds lookback 20 + fold t-stats). It has no artifact, no verifier, no reference, and no superseded marker. Crucially, it was NEVER executed, so no duplicate experiment entered the record — the lookback robustness was recorded once (37361000967).

## Verdict

**RETAIN (process decision RETAIN).** The minimal novelty guard already codified in LEARNING_STATE + LEARNING_EFFICIENCY is SUFFICIENT to prevent equivalent experiments: the durable history demonstrably prevented the one near-duplicate from being executed twice, and all queue variants have distinct objectives. The only residual gap is a single unmarked orphan code file (momentum_sweep.py) — a code-hygiene issue, not a research-process failure — recorded as the single minimal candidate task P-008 (mark superseded; comment-only; controller-validated). No process change is warranted.

## Change summary (what actually changed)

- `state/campaign/runs/46092417292/agents/agent_03.json` — audit record (single durable deliverable).
- `state/worker_progress.md` — semantic milestone (P-002 complete).
- `logs/ACTIVATION-2026-11-28.md` — this activation record.
- No research code, experiment, controller, or workflow file changed; no experiment rerun.

## Verification (inspection-only; no code change)

- Repo-wide `momentum_sweep` search: no artifact/verifier/markdown reference (orphan confirmed).
- state/check_artifacts/ listing: `momentum_sweep_results.json` absent; `momentum_lookback_sweep_results.json` present.
- LEARNING_STATE frontier/history: each frontier-first cell closed once; Learning rules #2 present.
- task_queue.json + campaign_status.json: R-001/R-004 RESOLVED; OPEN variants distinct; no repeated task_id.

## Unverified

- No live finish timestamp (bash/`date` denied); start from observed message time; none invented.
- Observational audit: no controlled counterfactual isolating the guard as the cause (confounded, as in P-001).
- Proposed P-008 change not executed; its independent controller validation is unmet here.
- Duplicate detection by docstring/objective + reference search, not by executing/diffing the two checks' numeric outputs.

## Next

Execute candidate task P-008: add a one-line "superseded by momentum_lookback_sweep.py + momentum_longer_horizon.py" note to the `research/checks/momentum_sweep.py` docstring (comment-only) and obtain independent controller validation that only a docstring changed.