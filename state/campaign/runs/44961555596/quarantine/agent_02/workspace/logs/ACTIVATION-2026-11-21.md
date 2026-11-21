# Activation Log — 2026-11-21

## Activation — run 44961555596, campaign slot 02 (global agent 2, task R-001)

**Environment observation:** observed message time 2026-11-21T23:28:42Z (the only
trusted wall-clock observation available at this layer). Run identity: GitHub
Actions run 44961555596, workflow "Kilo Research Sequential Learning Campaign",
attempt 1, ref main, repository root /home/runner/work/X/X. The `date` shell form
is denied at this layer, so no finish timestamp was captured; none was invented.

**Objective (task R-001):** Does the short-horizon momentum edge (lookback 3/5/10,
hold 1 day, daily rebalance) survive conservative transaction-cost stress on the
collected 10-asset universe, judged against a cost-matched coin-flip null?

**Selection basis:** highest evidence-backed qualitative priority among OPEN tasks
for this role; the momentum cell was the only admitted positive candidate and its
cost robustness was the documented open qualification.

**A-priori prediction (recorded before execution):** because the rule is a
high-turnover daily-rebalanced +/-1 exposure, realistic costs were expected to
remove a material part of the gross edge; the universe classification was expected
to be MIXED or COST_FRAGILE rather than COST_ROBUST. COST_ROBUST would refute the
gross-artifact concern.

**Predeclared cost grid (fixed before execution):**
- zero: `BacktestConfig()` (replicates the existing zero-cost path)
- realistic: commission_per_trade 2.0, commission_per_share 0.003,
  slippage_cents 2.0, slippage_proportional 0.0005 (the repository's existing
  realistic model from `examples/ma_crossover_real_data.py`)
- conservative: 5.0 / 0.005 / 5.0 / 0.0015 (~2.5x-3x the realistic model)

**Predeclared classification:** edge = REGIME_STABLE with all segment medians > 0;
COST_ROBUST = edge at zero + realistic + conservative; COST_SENSITIVE = edge at
zero + realistic, lost at conservative; COST_FRAGILE = edge at zero, lost at
realistic; NO_EDGE = no zero-cost edge. Universe COST_ROBUST >= 5/10, MIXED >= 1
robust or sensitive, COST_FRAGILE 0 robust/sensitive and >= 1 fragile, else
NO_EDGE.

## Work performed (chronological)

1. Read durable state: LEARNING_STATE.md, STATE.md, task_queue.json,
   current_task.json, worker_progress.md, the momentum check/verifier and prior
   artifacts, `engine.py`, `regime_stability.py`, `perturbation.py`, `leakage.py`,
   and the controller validator. Confirmed the existing momentum artifacts were
   produced with the default zero-cost `BacktestConfig` and that the repository's
   realistic cost model lives in `examples/ma_crossover_real_data.py`.
2. Wrote `research/checks/momentum_cost_sensitivity.py` (predeclared grid and
   classification above; `stress_segments` receives the level's `BacktestConfig`
   so both candidate and coin-flip null are charged the same costs; zero-cost
   cross-consistency gate; determinism gate; artifact).
3. Ran the check: manifest 10/10 OK; preflight PASSED; leakage PASS on AMZN/JPM;
   artifact written; zero-cost column reproduced `momentum_lookback_sweep_results.json`
   for all 10 tickers x 3 lookbacks; determinism r1==r2.
4. Wrote `research/checks/verify_momentum_cost_sensitivity.py` with four gates:
   fresh recomputation (independent volatility blocks/segments/momentum signals +
   `walk_forward` + `noise_benchmark`), zero-cost cross-check, classification
   reproducibility, cost-model directionality audit, plus determinism.
5. Ran the verifier: all gates MATCH; the directionality audit showed the cost
   model is not directionally adverse.
6. Ran the regression suite: 168 tests, all passing.
7. Recorded the result in STATE.md, LEARNING_STATE.md, worker_progress.md, and
   this log.

## Results

- Lookback 3: COST_ROBUST=6, NO_EDGE=4 -> COST_ROBUST.
- Lookback 5: COST_ROBUST=5, COST_FRAGILE=1 (MSFT), NO_EDGE=4 -> COST_ROBUST.
- Lookback 10: COST_ROBUST=5, NO_EDGE=5 -> COST_ROBUST.
- All 30 (asset, lookback) cells: COST_ROBUST=16, NO_EDGE=13, COST_FRAGILE=1.
- Retention (median realistic or conservative segment median / zero-cost median)
  is 0.99-1.00 for every asset with a zero-cost edge.
- Zero-cost column cross-check vs `momentum_lookback_sweep_results.json`: MATCH
  for all 10 tickers x 3 lookbacks.
- Verifier Gate 1 (AMZN/JPM fresh walk_forward + noise_benchmark, all lookbacks and
  levels): MATCH. Gate 2 (zero-cost column): MATCH. Gate 3 (all 30 cost classes
  and 3 universe verdicts): MATCH. Determinism: identical.
- Regression suite: 168 tests, all passing.

## Key finding — the existing cost model is not directionally adverse

- Every fill is priced at `close + slippage_cents/100 + close *
  slippage_proportional`, for buys and sells alike (`engine.py`; the existing
  `tests/test_engine.py::test_short_side` asserts this contract). For a long/short
  rule this makes sell-side fills favorable.
- Two-bar round trip with `slippage_cents=2.0` only: long and short both end at
  exactly 1,000,000.000000 (initial equity). Slippage has zero net cost.
- AMZN full sample (lookback 5, 4451 fills over 4465 bars): zero +5.537258;
  slippage-only +5.539447; commission-only +5.516203; realistic +5.518392.
- The "conservative" grid level scales a non-adverse slippage charge and a
  negligible commission, so it is not conservative for this strategy. The
  COST_ROBUST label is an upper bound under the existing model.
- MSFT lookback 5 COST_FRAGILE is a null-dispersion artifact: candidate medians
  are 0.051992 / 0.051945 / 0.051929 across zero/realistic/conservative (unchanged
  to four decimals) while the coin-flip null dispersion changes 0.008121 ->
  0.003300, flipping the verdict. This is not a cost effect.

## Verdict

**UNVERIFIED for conservative-cost robustness; COST_ROBUST (verified) only under
the repository's existing cost model.** The narrow cost grid was evaluated and
independently verified: the momentum edge is retained under the existing model at
all three lookbacks (5-6/10 assets, retention 0.99-1.00). But the verifier's
directionality audit shows the model's sell-side slippage is favorable, so the
grid does not constitute conservative stress. The a-priori prediction
(MIXED/COST_FRAGILE) was refuted for a cost-model reason, not because a robust
net edge was demonstrated. Momentum remains admitted as candidate positive
evidence with an explicit "cost robustness UNVERIFIED" qualification.

## Change summary (what actually changed)

- `research/checks/momentum_cost_sensitivity.py` — new check (predeclared cost
  grid, cost-matched null, classification, cross-consistency and determinism
  gates, artifact).
- `research/checks/verify_momentum_cost_sensitivity.py` — new independent
  verifier (four gates + determinism).
- `state/check_artifacts/momentum_cost_sensitivity_results.json` — new artifact.
- `state/STATE.md` — new activation section.
- `state/LEARNING_STATE.md` — frontier row, history row, Next action updated.
- `state/worker_progress.md` — phase/status/milestone.
- `logs/ACTIVATION-2026-11-21.md` — this record.

## Verification (exact post-change checks)

- `python3 research/checks/momentum_cost_sensitivity.py` — exits 0; manifest 10/10;
  preflight PASSED; leakage PASS; zero-cost cross-consistency MATCH; determinism
  r1==r2; artifact written.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` — Gates 1-3 MATCH;
  directionality audit completed; determinism identical.
- `python3 -m unittest discover -s tests` — 168 tests, all passing.

## Unverified

- Whether the momentum edge survives genuinely conservative (directionally adverse)
  transaction costs: the existing engine cannot price sell-side fills adversely,
  so this activation cannot answer the primary question.
- Finish timestamp: no UTC clock source available at this layer.
- The cost classification inherits the regime verdict's null-dispersion
  sensitivity (MSFT lookback-5 case); it is not a pure cost metric.
- Controller-side validation of the agent record was not run at this layer; it
  remains the deciding check.

## Next

Repair `research/backtest/engine.py` so buys fill at close + slippage and sells /
short-covers fill at close - slippage, add unit tests for symmetric adverse fills,
then re-run the same predeclared cost grid and verifier before any cost-robustness
claim.