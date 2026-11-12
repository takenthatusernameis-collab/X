# Activation Log — 2026-11-12

## Activation — 43181285853 / Global Agent 2 (campaign slot 02, task R-001)

**Observed timestamps (UTC):** activation start 2026-11-12T08:16:42Z (message time); log written 2026-11-12T08:25:25Z (observed via Python `datetime`). Intermediate checkpoint times were not individually captured.

**Environment observation:** GitHub Actions Linux runner, fresh Kilo Code CLI process, run_id 43181285853, workflow "Kilo Research Sequential Learning Campaign", repository root /home/runner/work/X/X, research/simulation only.

**Objective (task R-001):** Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe? Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null.

**Selection basis:** HIGHER_ORDER_OBJECTIVE; the frontier's Next action explicitly called for a conservative transaction-cost sensitivity test (fixed cost model, lookback 3/5/10, matched null) as the remaining decision-relevant qualification of the admitted momentum evidence.

**A-priori prediction (recorded before the run):** because constant-dollar sizing rebalances only the change in notional, realized costs should be small relative to the per-fold signal, so the edge is expected to survive realistic and conservative cost; a COST_FRAGILE outcome would instead show the admitted momentum evidence is an execution-cost artifact.

## Work performed (chronological)

1. **Inspected durable evidence:** LEARNING_STATE.md, STATE.md, campaign task_queue.json, agent_02_contract.json, worker_progress.md, the existing momentum checks (momentum.py, momentum_lookback_sweep.py) and their verifiers, and the framework cost model (engine.py, examples/ma_crossover_real_data.py).
2. **Diagnostic probe (scratch):** measured one `stress_segments` call (~0.15 s) and confirmed the realistic cost model is applied; used only to size the run and confirm the cost path, not to set the classification rule.
3. **Main check:** `research/checks/momentum_cost_sensitivity.py` — predeclared cost grid {zero, realistic, conservative=2x realistic} x lookback {3,5,10} x 10 tickers; manifest 10/10 OK, PREFLIGHT PASSED, leakage PASS on AMZN/JPM (signal integrity + equity-fill audit); per-segment candidate and matched coin-flip null medians via `stress_segments` under each cost model; full-sample total return and realized cost; predeclared survival classification; determinism r1==r2; artifact written.
4. **Independent verification:** `research/checks/verify_momentum_cost_sensitivity.py` — Gate 1 fresh `walk_forward` + `noise_benchmark` recomputation (AMZN/JPM, all cost levels and lookbacks) MATCH; Gate 2 zero-cost lookback-5 column cross-checks `momentum_lookback_sweep_results.json` exactly for all 10 tickers (medians and nulls); Gate 3 survival flags and universe classifications reproduce; Gate 4 determinism identical.
5. **Regression suite:** `python3 -m unittest discover -s tests -v` — 168 tests, all passing.
6. **Record:** durable JSON handoff written to `state/campaign/runs/43181285853/agents/agent_02.json`; `state/worker_progress.md` updated to AGENT_02_COMPLETE.

## Results

- **Classification: COST_ROBUST at every cost level and lookback.** 9/10 assets retain a strictly positive net-of-cost per-segment median at zero, realistic, and 2x conservative cost for lookbacks 3/5/10. NVDA is the sole non-survivor and was already non-positive at zero cost (cost-fragile outlier).
- **Realized full-sample cost drag (mean over 10 assets):** lookback 3 — zero 4.4739, realistic 4.4243 (drag 0.0497, ~1.1%), conservative 4.3746 (drag 0.0993, ~2.2%); lookback 5 — zero 4.4902, realistic 4.4406 (drag 0.0496), conservative 4.3910 (drag 0.0993); lookback 10 — zero 4.5091, realistic 4.4600 (drag 0.0491), conservative 4.4109 (drag 0.0982).
- **Retained net-of-cost median-segment fraction (mean/min over assets):** realistic 0.964/0.645 (lb3), 0.964/0.647 (lb5), 0.965/0.650 (lb10); conservative 0.930/0.305 (lb3), 0.930/0.309 (lb5), 0.879/-0.206 (lb10). The minimum is NVDA.
- **Mechanism:** constant-dollar sizing rebalances only the change in notional, so per-bar traded notional is a small fraction of the position and realized costs are small relative to the per-fold signal.

## Verdict

**RETAIN (cost-robust).** The short-horizon momentum edge survives conservative transaction-cost stress on the collected universe. The cost qualification that motivated task R-001 is closed; the momentum admission remains qualified as cost-robust, with NVDA flagged as a cost-fragile asset.

## Change summary

- `research/checks/momentum_cost_sensitivity.py` — new check (predeclared cost grid, matched-null regime gate, full-sample cost drag, survival classification, determinism, artifact).
- `research/checks/verify_momentum_cost_sensitivity.py` — new independent verifier (4 gates).
- `state/check_artifacts/momentum_cost_sensitivity_results.json` — new artifact.
- `state/campaign/runs/43181285853/agents/agent_02.json` — durable agent handoff.
- `state/worker_progress.md` — phase/status/milestone.
- `scratch/probe_cost*.py`, `scratch/analyze_cost.py`, `scratch/now.py`, `scratch/validate_record.py` — retained diagnostics.

## Verification (exact post-change commands)

- `python3 research/checks/momentum_cost_sensitivity.py` — exit 0; manifest 10/10; preflight PASSED; leakage PASS; determinism r1==r2; artifact written.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` — Gates 1-4 all MATCH/identical.
- `python3 -m unittest discover -s tests -v` — 168 tests, all passing.
- `python3 -m py_compile research/checks/momentum_cost_sensitivity.py research/checks/verify_momentum_cost_sensitivity.py` — compiles.
- `python3 scratch/validate_record.py` — agent_02.json passes the controller record schema (`record_failures: []`); candidate task R-006 valid.

## Unverified / caveats

- The framework coin-flip null is not turnover-matched; candidate-vs-null "beats" counts fell at conservative cost (7-8/10) while absolute survival stayed 9/10. The absolute net-of-cost medians and full-sample drag are the operative evidence; a turnover-matched null is recorded as candidate task R-006.
- The conservative level is a bounded 2x stress, not an exhaustive cost frontier; the break-even cost multiple was not mapped.
- The per-fold log-return metric has a mild fold-base sensitivity; it was cross-checked against the unambiguous full-sample drag and the two agree.
- No live or production execution was performed; research/simulation only.