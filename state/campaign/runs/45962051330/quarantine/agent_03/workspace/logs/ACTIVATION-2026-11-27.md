# Activation 2026-11-27 — Momentum transaction-cost sensitivity (R-001)

- Campaign run: 45962051330 (Kilo Research Sequential Learning Campaign)
- Agent: global agent 2, campaign slot 2
- Role: HIGHER_ORDER_OBJECTIVE
- Task: R-001
- Repository SHA: 618539a4a9627acf46f68bc72b855293fd64c871 (ref main)
- Start (UTC): 2026-11-27T08:03:45Z (environment message time)
- Finish (UTC): 2026-11-27T08:12:17Z (Python datetime probe; `date` shell form denied)

## Objective

Run a bounded, predeclared transaction-cost sensitivity check for lookback
3/5/10 momentum under the repository's existing realistic cost model and a
matched coin-flip null on the collected 10-asset adjusted-close universe, and
classify cost robustness with an a-priori prediction, independent recomputation,
determinism, and the existing regression suite.

Primary question: does the short-horizon momentum edge survive conservative
transaction-cost stress on the collected real-data universe?

## Why this task (pre-work reasoning)

1. The system is trying to get better at choosing, testing, falsifying, and
   preserving quantitative-research evidence under uncertainty.
2. The binding constraint is that the only positive candidate (short-horizon
   momentum) is admitted but its execution-cost robustness is unverified.
3. The uncertainty blocking a better decision is whether realistic costs erase
   the edge (tradable net of costs vs gross-only artifact).
4. The single bounded action is the predeclared cost grid through the existing
   `stress_segments` + matched null, then independent verification.
5. Worth it: LOW complexity/risk (reuses the framework, seconds of compute),
   HIGH information gain (cost-classifies the only positive candidate), HIGH
   falsification power.

## CHANGED

- `research/checks/momentum_cost_sensitivity.py` (new check).
- `research/checks/verify_momentum_cost_sensitivity.py` (new independent verifier).
- `state/check_artifacts/momentum_cost_sensitivity_results.json` (new artifact).
- `state/STATE.md`, `state/LEARNING_STATE.md` (frontier + history + Next).
- `state/activation_status.json`, `state/worker_progress.md`.
- `scratch/*_02.py` diagnostic helpers (retained per persistence policy).

No framework files (`research/backtest/**`), tests, examples, or protected
control-plane files were modified.

## Method (predeclared, fixed a-priori)

- Lookbacks {3, 5, 10}; cost levels {zero, realistic, stress2x}.
  - realistic = commission_per_trade=2.0, commission_per_share=0.003,
    slippage_cents=2.0, slippage_proportional=0.0005 (the repository's model).
  - stress2x = 2x every realistic component (conservative stress bound).
- Walk-forward per segment train=252d/test=84d/warmup=60d/overlap=60d; 4
  contiguous volatility blocks; min 400 bars/segment; each (asset, lookback,
  cost) candidate vs a coin-flip null run under the SAME cost cfg.
- A-priori per-asset classification: NO_BASELINE_EDGE if n_edge[zero]==0;
  COST_ROBUST if n_edge[zero]>=2 and n_edge[realistic]>=2; COST_FRAGILE if
  n_edge[realistic]==1; COST_ERASES_EDGE if n_edge[realistic]==0.
- A-priori prediction: COST_ROBUST for the six admitted assets; falsified if
  realistic costs cut positive-REGIME_STABLE lookbacks to <2 for a majority of
  admitted assets.

## VERIFIED (post-change commands that succeeded)

- `python3 research/checks/momentum_cost_sensitivity.py` — exits 0; manifest
  10/10 OK; PREFLIGHT PASSED; leakage PASS (AMZN/JPM, realistic costs); cost
  grid computed for all 3 levels x 3 lookbacks x 10 assets; classification
  COST_ROBUST=6 (AAPL, MSFT, GOOGL, AMZN, META, TSLA) / NO_BASELINE_EDGE=4
  (NVDA, JPM, JNJ, XOM) / COST_FRAGILE=0 / COST_ERASES_EDGE=0; cost drag at
  lookback 5 = 0.0000 for all but NVDA (+0.009); determinism r1==r2 True;
  zero-cost column cross-consistent with
  `momentum_lookback_sweep_results.json` True; artifact written.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` — Gate 1 PASS
  (fresh walk_forward + noise_benchmark recomputation, AMZN/JPM, all
  lookbacks/costs, MATCH); Gate 2 PASS (zero-cost column reproduces the
  lookback sweep on all 10 assets / 3 lookbacks); Gate 3 PASS (classification
  re-derives, cost params match, AMZN lb5 realistic determinism identical).
- `python3 -m unittest discover -s tests -q` — 168 tests, all passing (OK).
- `git diff --check` clean on touched files.

## UNVERIFIED / caveats

- Exact per-checkpoint wall-clock beyond the session start and a single finish
  probe; `date` is denied so no independent clock cross-check.
- The n_edge classification is null-dispersion-sensitive (the coin-flip null
  trades far more than the low-turnover signal, so its dispersion changes under
  costs); n_edge can shift +/-1 with no candidate-median change (e.g. MSFT lb5
  zero=REGIME_STABLE vs realistic=REGIME_DEPENDENT on identical medians
  [0.052, 0.072]). The candidate-median invariance (cost drag ~0) is the
  primary evidence.
- Robustness is conditional on LOW TURNOVER (~1 sign flip/asset over 2009-2026;
  persistent long bias). Not a stringent test of a high-turnover timing alpha;
  does not establish live tradability. Exploratory research simulation only.

## Result

COST_ROBUST. The admitted short-horizon momentum edge survives the realistic
cost model and a 2x stress with essentially unchanged candidate medians for
exactly the six previously-admitted assets; 4/10 have no baseline edge; 0
COST_FRAGILE / 0 COST_ERASES_EDGE. Costs are genuinely charged but small
relative to the edge. Independently verified (3 gates MATCH) and deterministic.

## NEXT

Run R-002 (predeclared holding-period set for lookback-5 momentum), then R-003
(single past-only volatility gate), before revisiting a genuinely new signal
class.