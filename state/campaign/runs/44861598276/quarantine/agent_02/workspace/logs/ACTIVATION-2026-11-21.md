# Activation Log — 2026-11-21

## Activation — 44861598276 / agent 02 (campaign slot 02, R-001)

**Environment observation:** message time 2026-11-21T03:36:40Z (runner wall
clock); observed UTC checkpoints 2026-11-21T03:45:52Z (mid-run) and this record
written at activation finish (~2026-11-21T03:5xZ). Run identity
GITHUB_RUN_ID=44861598276, GITHUB_RUN_ATTEMPT=1,
event SHA=618539a4a9627acf46f68bc72b855293fd64c871, REF_NAME=main, repository
root=/home/runner/work/X/X. Research mode: simulation only.

**Pre-work notes (required):**

1. *What is the system ultimately trying to become better at learning?* It is
   trying to choose, test, falsify, independently validate, and preserve
   quantitative-research hypotheses so that durable uncertainty falls per unit
   of worker effort.
2. *What currently most constrains useful learning?* The only admitted positive
   candidate (short-horizon momentum) is mechanical and daily-rebalanced; its
   execution-cost robustness is the last decision-relevant qualification before
   it can be treated as more than a simulation artifact.
3. *What uncertainty prevents a better decision?* Whether the momentum segment
   medians survive realistic transaction costs, and whether the repository's
   cost model is conservative enough for that conclusion to mean anything.
4. *What single bounded action is most likely to reduce it?* Run lookback 3/5/10
   momentum through the existing regime-stability gate under a predeclared cost
   grid (zero / realistic / 2x stress) with a matched coin-flip null, and
   independently verify the artifact.
5. *Why is that action worth its effort, complexity, and execution risk?* It is
   low-complexity (reuses the existing engine/gate/null), high falsification
   power (cost elimination would overturn the candidate), and directly decision
   relevant; execution risk is low because it is read-only research simulation
   on already-collected data.

**Objective (task R-001):** Run a bounded cost-sensitivity check for lookback
3/5/10 momentum under the repository's existing realistic cost model and matched
null.

**Primary question:** Does the short-horizon momentum edge survive conservative
transaction-cost stress on the collected real-data universe?

**Selection basis:** highest evidence-backed qualitative priority among OPEN
HIGHER_ORDER_OBJECTIVE tasks (compared against R-001, R-002, R-003, R-005).

## Work performed (chronological)

1. **Read durable evidence:** `state/LEARNING_STATE.md` (active delta, frontier
   table, learning rules), `state/STATE.md` (frontier sections), `state/STATE.md`
   cost references, `state/campaign/current_task.json` (contract),
   `state/campaign/task_queue.json` (R-001), `state/worker_progress.md`, the
   existing momentum checks (`research/checks/momentum.py`,
   `momentum_lookback_sweep.py`, `momentum_longer_horizon.py`) and verifiers,
   `research/backtest/engine.py`, `regime_stability.py`, `perturbation.py`,
   `real_data.py`, `leakage.py`, and the existing cost model in
   `examples/ma_crossover_real_data.py` / `ma_crossover.py` /
   `volatility_regime_filter.py`.

2. **Predeclared the bounded cost grid (before execution):** zero (repository
   default), realistic ($2/trade + $0.003/share, 2c fixed + 5 bps proportional
   slippage), stress (2x realistic). Predeclared the classification rules
   (per-asset and aggregate) in the check docstring before running.

3. **New check:** `research/checks/momentum_cost_sensitivity.py` — manifest
   integrity, preflight gate, leakage review at all three cost levels
   (AMZN/JPM), full 3 lookbacks x 3 cost levels regime-stability sweep with a
   matched coin-flip null, per-asset and aggregate cost-robustness
   classification, an explicit verdict-independent "strictly positive medians"
   view, determinism assertion, artifact written to
   `state/check_artifacts/momentum_cost_sensitivity_results.json`.

4. **Independent verifier:** `research/checks/verify_momentum_cost_sensitivity.py`
   — Gate 1 fresh `walk_forward` + `noise_benchmark` recomputation (no
   `stress_segments`) on AMZN/JPM for all lookbacks x cost levels; Gate 2
   zero-cost cross-consistency vs `momentum_lookback_sweep_results.json` on all
   10 tickers x 3 lookbacks; Gate 3 classification/median-positive-count
   reproducibility; Gate 4 cost-grid integrity; determinism.

5. **Records:** `state/STATE.md` (new activation section),
   `state/LEARNING_STATE.md` (frontier row + learning-history row + Next
   update), `state/worker_progress.md` (milestone), this log, and the durable
   agent record `state/campaign/runs/44861598276/agents/agent_02.json`.

## Results

Cost grid: zero / realistic / stress(2x), lookbacks 3/5/10, 10-asset universe,
walk-forward per volatility segment (train=252d/test=84d/warmup=60d/overlap=60d,
4 blocks, min 400 bars/segment), matched coin-flip null under the same costs.

- Candidate segment medians are essentially cost-invariant. At lookback 5,
  9/10 assets have strictly positive segment medians at zero, realistic, and
  2x stress; the only exception is NVDA (one negative segment that worsens with
  cost: -0.038 -> -0.064 -> -0.088).
- Verdict-based positive-edge counts at lookback 5: 6/10 at all three cost
  levels. Lookback-ROBUST (>=2 lookbacks with a positive edge): 6/10 at all
  three levels. Aggregate cost-robustness class: **COST_ROBUST**.
- Verdict flips without candidate degradation: MSFT medians unchanged
  [0.052, 0.072] but verdict REGIME_STABLE -> REGIME_DEPENDENT at realistic
  cost; JPM medians unchanged [0.113, 0.044, 0.083] but verdict
  REGIME_DEPENDENT -> REGIME_STABLE. These are matched-null dispersion
  artifacts; the verdict-independent median view is the reliable reading.
- Cost magnitude under this model is small: full-sample AMZN realistic
  round-trip cost $135,860 on $1M initial capital over 2009-2026; zero ->
  realistic total-return difference 0.019 (1.9% of initial). Cause: the engine
  adds slippage to the fill price for both buys and sells (symmetric, not
  adverse), so slippage largely cancels over round trips and commission
  dominates. The existing cost model is therefore optimistic for a high-turnover
  strategy.

## Verdict

**COST_ROBUST within the repository's existing cost model** (independently
verified): the momentum candidate's positive segment medians are not eliminated
by realistic or 2x stress costs, and the predeclared aggregate class is
COST_ROBUST. This resolves the recorded cost-robustness qualification for that
model. It does **not** settle adverse-slippage cost robustness: the engine's
symmetric slippage makes the test optimistic. The momentum candidate remains
admitted as candidate positive evidence with this qualification; no negative
result overturns it. The next decision-relevant check is an adverse-slippage
cost model (buy fills above close, sell fills below close) on the same universe.

## Change summary (what actually changed)

- `research/checks/momentum_cost_sensitivity.py` — new bounded cost-sensitivity
  check.
- `research/checks/verify_momentum_cost_sensitivity.py` — new independent
  verifier.
- `state/check_artifacts/momentum_cost_sensitivity_results.json` — new artifact.
- `state/STATE.md`, `state/LEARNING_STATE.md`, `state/worker_progress.md`,
  `logs/ACTIVATION-2026-11-21.md` — records.
- `state/campaign/runs/44861598276/agents/agent_02.json` — durable agent record.
- `scratch/probe_env.py`, `scratch/now.py`, `scratch/analyze_cost.py` — small
  execution/diagnostic helpers (retained per the persistence policy; not part of
  the research evidence).

No framework code (`research/backtest/**`), tests, examples, or control-plane
files were modified.

## Verification (exact post-change checks)

- `python3 research/checks/momentum_cost_sensitivity.py` — exits 0; manifest
  10/10 OK; preflight PASSED; leakage PASS at all three cost levels; artifact
  written; determinism r1==r2 True.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` — exits 0; Gate 1
  (AMZN/JPM, 3 lookbacks x 3 cost levels) MATCH; Gate 2 (zero-cost column vs
  `momentum_lookback_sweep_results.json`, 10 tickers x 3 lookbacks) MATCH; Gate 3
  (per-asset classification, median-positive counts, aggregate class,
  lookback-robust counts) MATCH; Gate 4 (cost-grid integrity) MATCH; determinism
  identical.
- `python3 -m unittest discover -s tests -v` — 168 tests, all passing.

## Unverified

- Adverse-slippage cost robustness: the engine applies slippage symmetrically
  (favorable on sells), so the realistic/stress results are optimistic and do
  not bound a direction-dependent (adverse) slippage model.
- The verdict-based per-asset classification is null-dispersion-sensitive
  (MSFT/JPM flips with unchanged medians); it is recorded but not treated as the
  primary reading.
- Only the repository's 10 large-cap assets and one cost model were tested; no
  claim is made about other universes, other cost regimes, or live execution.
- The 2x stress level is a bounded stress, not an extreme-cost bound.