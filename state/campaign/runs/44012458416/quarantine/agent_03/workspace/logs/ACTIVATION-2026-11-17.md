# Activation Log — 2026-11-17

## Activation — campaign slot 02 (task R-001, cost-sensitivity of the momentum candidate)

**Environment observation:** message time 2026-11-17T03:49:55Z (runner wall clock); activation start 2026-11-17T03:49:55Z; this record written 2026-11-17T04:02:38Z; run identity GITHUB_RUN_ID=44012458416, GITHUB_RUN_ATTEMPT=1, event SHA=618539a4a9627acf46f68bc72b855293fd64c871, REF=main, repository root=/home/runner/work/X/X. Execution layer: GitHub Actions Linux runner, fresh Kilo Code CLI process. Research/simulation only.

**Objective (task R-001):** Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null; decide whether the admitted short-horizon momentum candidate survives conservative transaction-cost stress on the collected real-data universe.

**Selection basis:** the contract's focused task. The candidate positive evidence in `LEARNING_STATE` was produced with the framework's DEFAULT (zero-cost) `BacktestConfig`; a mechanically daily-rebalanced 1-day bet is exactly the class whose verdict is decided by costs. Prior agent slot 01 produced no controller-validated durable record, so this activation proceeded on the durable contract.

**Predeclared cost grid (fixed before execution, not tuned):**
- `zero` — `BacktestConfig()` defaults.
- `realistic` — the repository's existing realistic model copied verbatim from `examples/ma_crossover_real_data.py` (commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005).
- `conservative_2x` — every realistic component doubled.

**A-priori falsification prediction (fixed before execution):** the momentum signal changes exposure when the lookback return changes sign, so it churns; the fixed 2-cents-per-share slippage is large relative to split-adjusted historical prices (e.g. AAPL 2009 adjusted close ~2.7). Prediction: candidate medians fall sharply at `realistic` and FEWER THAN HALF of zero-cost EDGE cells remain EDGE -> `COST_FRAGILE`. `COST_ROBUST` would refute the prediction.

## Work performed (chronological)

1. Read durable state: `LEARNING_STATE.md`, `STATE.md`, `worker_progress.md`, campaign `current_task.json` / `task_queue.json`, the momentum checks/artifacts, `research/backtest/{engine,perturbation,regime_stability}.py`, and the controller validation script. Baseline regression suite: 168/168 tests pass.

2. Added `research/checks/momentum_cost_sensitivity.py`: the predeclared 3x3 grid (cost level x lookback) over the 10-asset universe through the existing regime gate, with the matched coin-flip null costed identically (stress_segments forwards `cfg` to both candidate and null). Leakage + equity-fill audit under every cost level on AMZN/JPM. Wrote `state/check_artifacts/momentum_cost_sensitivity_results.json`.

3. The first run produced a surprising result (costs barely move medians; AMZN full-sample total return 5.5373 -> 5.5184). Rather than trust the `COST_ROBUST` headline, added a turnover / cost-incidence diagnostic (weight counts, sign flips, cost drag) because a cost verdict is uninterpretable if the signal does not take directional positions. This is an interpretability diagnostic, not a change to the predeclared grid or classification rule.

4. Added `research/checks/verify_momentum_cost_sensitivity.py` with five independent gates: (1) fresh `walk_forward` + `noise_benchmark` recomputation of AMZN/JPM medians/nulls for all lookback x cost cells; (2) independent re-derivation of edge_status / cost_robustness / counts / headline; (3) zero-cost column cross-consistency vs `momentum_lookback_sweep_results.json`; (4) determinism; (5) turnover/degeneracy reproducibility from a fresh signal implementation.

## Results

**Predeclared cost-robustness classification: `COST_ROBUST`** (survival_fraction 0.941; 16 of 17 zero-cost EDGE cells remain EDGE at `realistic`; 18 EDGE cells at `realistic`).

- Edge counts (30 cells = 10 assets x 3 lookbacks): zero EDGE=17 / LOSS=0 / NO_EDGE=13; realistic EDGE=18 / LOSS=0 / NO_EDGE=12; conservative_2x EDGE=18 / LOSS=0 / NO_EDGE=12.
- Verdict counts: zero {REGIME_STABLE=19, REGIME_DEPENDENT=5, CONSISTENT_WITH_NOISE=6}; realistic and conservative_2x {REGIME_STABLE=21, REGIME_DEPENDENT=3, CONSISTENT_WITH_NOISE=6}.
- Cost-robustness classes: SURVIVES=16, SURVIVES_REALISTIC=0, COST_FRAGILE=1 (MSFT lookback 5), NO_EDGE_AT_ZERO=13.

**Material qualification — the cost test is DEGENERATE (`degenerate_cost_test=true`):**

- Directional signal turnover is near zero: median 1 sign flip across ~4000 bars. 9/10 assets have `frac_long=1.00`, `mean_weight=+0.999`, i.e. the signal is long after the lookback warmup and never changes direction. NVDA has 2 flips (`frac_long=0.58`, `mean_weight=+0.169`), consistent with its split-adjusted price crossing $1.
- Source inspection: `research/backtest/regime_stability.py::momentum_signals` uses `ret = mean(log(closes[i-lookback+1:i+1]))` as its sign statistic. That is the mean log PRICE, not the mean log RETURN (a difference of log prices). For split-adjusted prices above $1 for the whole sample, `ret > 0` always, so the signal is constant-long. `mean_reversion_signals` is its exact opposite (constant-short).
- Consequence: the prior "momentum edge" (7/10 REGIME_STABLE positive) is consistent with long-only market exposure, not short-horizon return autocorrelation; the "reversal loses everywhere" result is constant-short exposure; and the previously reported -0.99 reversal/momentum P&L correlation is trivially true because the two signals are exact opposites.
- The cost verdict therefore does NOT validate an execution-realistic momentum trading edge: costs are not the binding constraint only because the candidate never changes direction. Mean cost drag is +0.0496 return units (NVDA +0.3536, driven by fixed per-share slippage on low split-adjusted prices; other assets ~0.01).

## Verdict

**Decision: VERIFIED_NEGATIVE_RESULT.** The predeclared `COST_FRAGILE` prediction was refuted (headline `COST_ROBUST`), but the independent turnover diagnostic shows the cost comparison is degenerate: the momentum candidate is constant-exposure (1 directional flip for 9/10 assets), so the cost question is not decision-relevant for a momentum strategy, and the admitted momentum evidence is confounded by a signal-construction defect (mean log price used as the sign statistic). This is durable, independently verified negative/qualifying evidence. It does NOT overturn any prior file; it re-characterizes the momentum evidence base and hands off the repair as the highest-value next action.

## Change summary (CHANGED / VERIFIED / UNVERIFIED / NEXT)

**CHANGED**
- `research/checks/momentum_cost_sensitivity.py` (new) — predeclared 3x3 cost x lookback sweep, matched costed null, leakage/equity audit, turnover/degeneracy diagnostic.
- `research/checks/verify_momentum_cost_sensitivity.py` (new) — five-gate independent verifier.
- `state/check_artifacts/momentum_cost_sensitivity_results.json` (new) — reproducible artifact.
- `scratch/hash_check.py` (new) — scratch diagnostic (UTC time + campaign-state hash integrity).
- `logs/ACTIVATION-2026-11-17.md` (new) — this record.
- `state/worker_progress.md` — semantic milestone update.

**VERIFIED (exact commands that ran after the final edits)**
- `python3 research/checks/momentum_cost_sensitivity.py` — exit 0; manifest 10/10 OK; preflight PASSED; leakage + equity-fill audit PASS at all three cost levels; 30 cells; headline `COST_ROBUST`; `degenerate_cost_test=true`; determinism r1==r2 True; artifact written.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` — exit 0; Gate 1 (AMZN/JPM x 3 lookbacks x 3 cost levels) all MATCH; Gate 2 classification MATCH; Gate 3 zero-cost cross-consistency all 30 cells MATCH vs `momentum_lookback_sweep_results.json`; Gate 4 determinism identical; Gate 5 turnover/degeneracy MATCH.
- `python3 -m unittest discover -s tests -q` — Ran 168 tests, OK.
- `python3 -m py_compile research/checks/momentum_cost_sensitivity.py research/checks/verify_momentum_cost_sensitivity.py` — clean.
- `git diff --check` — clean.

**UNVERIFIED**
- Whether a corrected log-return momentum signal would survive costs is untested (out of scope for this activation; see candidate task R-006).
- Whether the mean-log-price sign statistic was intentional rather than a defect is not established from durable evidence; the diagnosis is empirical (constant exposure) plus source inspection.
- Cost attribution between fixed-per-share and proportional components is not isolated (see candidate task R-007).

**NEXT (one bounded action)**
Audit `research/backtest/regime_stability.py::momentum_signals` (and its mirror `mean_reversion_signals`) against the turnover evidence, repair the sign statistic to a log return, and re-run the existing momentum frontier gate stack with independent verification.