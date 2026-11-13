# Activation Log — 2026-11-13

## Activation — run 43407543668 / campaign slot 02 (agent 02, task R-001)

**Environment observation:** message time 2026-11-13T10:32:26Z; controller `agent_02_before.json` captured 2026-11-13T10:32:20Z; observed finish 2026-11-13T10:45:37Z (UTC, via Python `datetime`); run_id 43407543668, run_attempt 1, ref main, repository root `/home/runner/work/X/X`. Research/simulation only.

**Objective (task R-001):** Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null; classify whether the qualified short-horizon momentum edge survives conservative transaction-cost stress on the collected 10-asset real-data universe.

**A-priori classification rule (fixed before execution):** per (asset, lookback, cost) EDGE if the segment verdict is REGIME_STABLE and every segment median is positive; COST_ROBUST if >=2 of 3 lookbacks are EDGE, COST_FRAGILE if 1, COST_ERASED if 0; universe PERSISTS if every zero-cost COST_ROBUST asset stays COST_ROBUST, PARTIAL if a non-empty subset, ERASED if none.

**Predeclared cost grid:** zero (defaults); realistic ($2/trade + $0.003/share, 2c fixed, 5 bps proportional — the repository's documented model); conservative (same commission, 4c, 10 bps); stress (same commission, 8c, 20 bps). Applied identically to the candidate and the matched coin-flip null.

## Work performed

1. **Readiness:** inspected LEARNING_STATE.md, STATE.md, campaign contract/status/queue, agent_01 record, the existing momentum checks, engine, perturbation, regime-stability and the prior lookback artifact. Baseline regression suite: 168/168 tests pass.

2. **New check:** `research/checks/momentum_cost_sensitivity.py` — manifest 10/10 OK, preflight PASSED (57 checks), leakage PASS on AMZN/JPM with costs (signal integrity + fill-equity audit), full 4x3x10 grid, Gate 0 zero-cost cross-consistency vs `momentum_lookback_sweep_results.json`, determinism r1==r2, artifact written to `state/check_artifacts/momentum_cost_sensitivity_results.json`.

3. **Independent verifier:** `research/checks/verify_momentum_cost_sensitivity.py` — Gate 1 fresh AMZN/JPM `walk_forward` + `noise_benchmark` recomputation at every lookback x cost MATCH; Gate 2 zero-cost column vs prior lookback artifact MATCH for all 10 assets; Gate 3 robustness + universe classification reproduced from full-precision artifact values MATCH; Gate 4 cost-drag reproduced MATCH; Gate 5 cost-model behavior reproduced and slippage-only >= zero confirmed.

4. **Characterization tests:** `tests/test_cost_model.py` — 3 tests pinning the engine's cost accounting (fixed slippage cancels over a round trip; proportional slippage scales P&L by (1+p); commission is the only true round-trip cost).

5. **Regression suite:** 171/171 tests pass (168 existing + 3 new).

## Results

- Cost grid: 6/6 zero-cost COST_ROBUST assets (AAPL, MSFT, GOOGL, AMZN, META, TSLA) remain COST_ROBUST at realistic, conservative and stress levels -> universe verdict PERSISTS at every level.
- Median cost drag vs zero cost is 0.0000 (range -0.003..+0.001), i.e. the grid removes essentially no return.
- AMZN full-sample diagnostic (lookback 5): zero +5.5373; commission-only +5.5162; slippage-only-stress +5.5460 (higher than zero); full-stress +5.5250. Commission paid $21,055.22 over 4451 fills; slippage paid $0 commission.
- Mechanism: the engine applies the same additive fill price (`close + slippage_cents/100 + close*slippage_proportional`) on entry and exit, so fixed slippage cancels over a completed round trip and proportional slippage multiplies both legs by (1+p); fold `total_return` is `eq[-1]/eq[0]-1` on the test segment, so costs incurred before the test segment are excluded.

## Verdict

**UNVERIFIED (cost robustness).** The bounded check is reproducible and independently verified, and under the existing model the momentum edge is cost-robust (6/6 baseline assets survive every predeclared level). But the existing "realistic cost" model does not charge a genuine transaction cost: slippage is not a round-trip cost in this engine. The check therefore does not establish robustness to external transaction costs. The durable record preserves this as UNVERIFIED with the exact mechanism and the next executable step.

## Change summary

- `research/checks/momentum_cost_sensitivity.py` — new check (predeclared 4-level cost grid, matched null, Gate 0, determinism, cost-drag and cost-model diagnostics).
- `research/checks/verify_momentum_cost_sensitivity.py` — new independent verifier (5 gates).
- `tests/test_cost_model.py` — new characterization tests.
- `state/check_artifacts/momentum_cost_sensitivity_results.json` — new artifact.
- `state/LEARNING_STATE.md` — frontier row, Next action, learning-history row.
- `state/activation_status.json` — updated for this activation.
- `state/worker_progress.md` — milestone update.
- `state/campaign/runs/43407543668/agents/agent_02.json` — durable agent record.

## Verification (exact post-change checks)

- `python3 -B research/checks/momentum_cost_sensitivity.py` — exits 0; manifest 10/10; PREFLIGHT PASSED; leakage PASS; Gate 0 MATCH; determinism r1==r2; artifact written.
- `python3 -B research/checks/verify_momentum_cost_sensitivity.py` — all 5 gates MATCH; prints "All independent recomputations MATCH ...".
- `python3 -m unittest tests.test_cost_model -v` — 3 tests OK.
- `python3 -m unittest discover -s tests -q` — 171 tests, OK.

## Unverified / caveats

- Cost robustness to *external* transaction costs is UNVERIFIED; the engine's slippage accounting must be repaired first.
- Per-checkpoint UTC timestamps other than start (message time) and finish (observed via Python) were not captured.
- The verdict-flip cases (MSFT REGIME_STABLE -> REGIME_DEPENDENT at realistic; JPM -> REGIME_STABLE; JNJ -> COST_ROBUST at stress) are driven by the null also paying the same (non-)costs and by the multiplicative proportional-slippage artifact; they do not indicate a genuine cost effect.

## Next

Repair the engine's cost accounting so slippage is charged directionally (buy above mid, sell below mid) with a regression test, then rerun `research/checks/momentum_cost_sensitivity.py` and its verifier to obtain a genuine cost-robustness classification.