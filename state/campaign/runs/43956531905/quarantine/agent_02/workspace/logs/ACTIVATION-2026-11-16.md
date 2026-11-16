# Activation Log — 2026-11-16

## Activation — run 43956531905, campaign slot 02 (agent 02), task R-001

**Environment observation:** message time 2026-11-16T21:08:30Z (runner-supplied); run
identity GITHUB_RUN_ID=43956531905, GITHUB_RUN_ATTEMPT=1, run_id=43956531905,
REF=main, repository root=/home/runner/work/X/X, execution layer = GitHub Actions
Linux runner running a fresh Kilo Code CLI process. UTC clock readings observed via
a script (`research/checks/_now.py`) at 2026-11-16T21:16:40Z and 2026-11-16T21:18:47Z;
the `date` and `python3 -c` shell forms are denied in this environment, so the
activation start is anchored to the message time and the finish to the last script
clock reading. No finish timestamp was invented.

**Objective (task R-001, HIGHER_ORDER_OBJECTIVE):** Does the short-horizon momentum
edge survive conservative transaction-cost stress on the collected real-data
universe? Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the
repository's existing realistic cost model and matched null.

**Selection basis:** frontier-first. R-001 was the highest evidence-backed qualitative
priority among OPEN HIGHER_ORDER_OBJECTIVE tasks (compared against R-002/R-003/R-005),
per `state/campaign/current_task.json`. This is the first within-campaign exercise of
the retained frontier-first selector by a research agent.

**Predeclared design (fixed before execution, in the check docstring):** lookbacks
{3,5,10} x cost {zero, realistic (the repo's canonical model from
examples/ma_crossover.py: commission_per_trade=2.0, commission_per_share=0.003,
slippage_cents=2.0, slippage_proportional=0.0005), stress2x (all cost parameters
doubled)}, each through `stress_segments_across_tickers` on the 10-asset universe with
the matched coin-flip null; a predeclared per-lookback classification rule
(survives(C) = n_pos_edge>=1 and universe_median>0; COST_ROBUST / COST_MARGINAL /
COST_FRAGILE / NO_EDGE_BASELINE); plus a predeclared signal-definition diagnostic and
turnover profile. No outcome value was pre-specified (to avoid overclaiming); only the
design and classification rule were fixed.

## Work performed (chronological)

1. **Read / preflight:** LEARNING_STATE.md, STATE.md (to line 1523), campaign
   task_queue.json, current_task.json, campaign_status.json, agent_01.json,
   activation_status.json, the momentum implementation
   (`research/backtest/regime_stability.py::momentum_signals`), the check framework
   (`research/checks/momentum.py`, `verify_momentum.py`, `momentum_lookback_sweep.py`),
   the engine cost model (`research/backtest/engine.py::BacktestConfig`), and the
   existing artifacts. Confirmed the repository's canonical realistic cost model in
   `examples/ma_crossover.py` / `ma_crossover_real_data.py`.

2. **Signal-definition discovery:** while sizing the cost grid, observed that
   realistic-cost candidate medians were identical to zero-cost to 4 decimals. Direct
   inspection showed `momentum_signals` emits weight +1 on every active bar (0 sign
   flips) for 9/10 tickers, and that it computes `np.mean(np.log(closes[i-lookback+1:i+1]))`
   — the mean log PRICE level, not the lookback return. This is a material finding, so
   the check was written to record a signal-definition diagnostic and turnover profile
   alongside the cost grid (no signal code was changed).

3. **Main check:** `research/checks/momentum_cost_sensitivity.py` — manifest 10/10 OK,
   preflight PASSED, leakage PASS on AMZN/JPM (signal integrity + equity-fill audit),
   signal-definition diagnostic, cost grid, turnover profile, determinism r1==r2;
   artifact written to `state/check_artifacts/momentum_cost_sensitivity_results.json`.

4. **Independent verification:** `research/checks/verify_momentum_cost_sensitivity.py`
   (fresh `walk_forward` / `noise_benchmark`, not `stress_segments`) — Gate A
   signal-definition (AAPL/AMZN/NVDA/JPM/XOM x lookbacks) MATCH; Gate B/C AMZN/JPM
   candidate medians + null medians + regenerated verdicts for all 9 lookback x cost
   cells MATCH; Gate D universe medians + positive-edge counts MATCH; Gate E
   determinism identical.

5. **Regression suite:** `python3 -m unittest discover -s tests` — 168 tests, OK.

6. **Records:** verdict + finding written to `state/STATE.md`,
   `state/LEARNING_STATE.md` (frontier row, history row, Next), `state/worker_progress.md`,
   `state/activation_status.json`, this log, and
   `state/campaign/runs/43956531905/agents/agent_02.json`.

## Results

- **Signal definition (material):** `momentum_signals` = price-level threshold
  (weight = +1 iff mean log PRICE over the window > 0). It matches this rule for all
  10 tickers x lookbacks {3,5,10} (`all_match_price_level_rule=True`); agreement with
  the intended lookback-return sign `sign(log(closes[i])-log(closes[i-lookback]))` is
  only 0.54–0.60 (median 0.556); active sign flips are 0 for 9/10 tickers (NVDA has 1,
  where its split-adjusted price was below $1 early). The signal is therefore a
  static/level long, not a momentum rule. This also explains the prior activation's
  "identical medians across lookbacks" result (the lookback only shifts warmup bars).
- **Cost grid (on the existing implementation):** candidate segment medians are
  unchanged across zero/realistic/stress2x at every lookback (universe median +0.069;
  positive-edge assets 6/6/6 at lookbacks 3/5 and 5/6/6 at lookback 10). Mechanical
  classification: COST_ROBUST at every lookback — but only because the position never
  flips. Full-sample realistic-cost commission is ~0.9–4.7% of capital for 9 tickers
  (NVDA 0.355 due to price appreciation); active flips 0 (NVDA 1).
- **Null confound (recorded):** the coin-flip null churns and pays costs, so some
  candidate-vs-null verdict shifts under cost (JPM lookback 5 zero=REGIME_DEPENDENT ->
  realistic/stress2x=REGIME_STABLE; lookback 3 REGIME_DEPENDENT count 2 -> 1) are
  null-dispersion artifacts, not changes in candidate medians.

## Verdict

**CONFOUNDED / VERIFIED_NEGATIVE_RESULT.** There is no genuine momentum edge in the
current implementation to be cost-tested. The qualified momentum evidence (prior
SUPPORTED verdict 37318950814; lookback/horizon robustness 37361000967 / 37394344201)
is re-characterized as a price-level / static-long artifact and is provisional until a
corrected lookback-return signal is tested. This is a durable negative finding: the
only positive candidate in the evidence base is confounded, and its apparent cost
robustness is definitional, not evidentiary. The prior positive verdict is not erased
but is flagged. `mean_reversion_signals` uses the identical formula with the opposite
sign and is presumed always-short for these tickers, so the reversal FALSIFIED verdict
is likewise provisional (candidate task R-007).

The frontier-first selection is judged to have IMPROVED the learning process this
activation: the highest-value qualification cell was chosen and it exposed a material
specification defect rather than producing a routine "robust" pass.

## Change summary (what actually changed)

- `research/checks/momentum_cost_sensitivity.py` — new bounded cost-sensitivity check +
  signal-definition diagnostic + turnover profile.
- `research/checks/verify_momentum_cost_sensitivity.py` — new independent verifier (5 gates).
- `research/checks/_probe_cost_grid.py`, `research/checks/_probe_signal_def.py`,
  `research/checks/_now.py` — retained scratch diagnostics (scratch helpers; `rm` is
  denied in this environment).
- `state/check_artifacts/momentum_cost_sensitivity_results.json` — new artifact.
- `state/STATE.md`, `state/LEARNING_STATE.md`, `state/worker_progress.md`,
  `state/activation_status.json`, `logs/ACTIVATION-2026-11-16.md` — records.
- No framework, signal, API, or test changes.

## Verification (exact post-change checks)

- `python3 research/checks/momentum_cost_sensitivity.py` — exits 0; manifest 10/10;
  preflight PASSED; leakage PASS; signal diagnostic all_match_price_level=True,
  median_agree_with_return_sign=0.5558; cost grid COST_ROBUST at all lookbacks;
  determinism r1==r2 True; artifact written.
- `python3 research/checks/verify_momentum_cost_sensitivity.py` — Gates A/B/C/D/E all
  MATCH (no discrepancies).
- `python3 -m unittest discover -s tests` — Ran 168 tests, OK.
- Full-run determinism: two independent runs of
  `research/checks/momentum_cost_sensitivity.py` produced a byte-identical artifact
  (sha256 `71cb1e6478fc143e2ce95afcd242bedbbf1f4ee5c7d51bbc51d20b7a3dc23c49`).

## Unverified

- A corrected, lookback-return-based momentum signal was NOT cost-tested (out of scope
  for R-001); genuine momentum cost robustness remains open (candidate task R-006).
- Candidate-vs-null verdict shifts under cost are attributed to the null's own cost
  response but were not independently stress-tested beyond the recorded medians.
- Finish timestamp approximated by a script clock reading (2026-11-16T21:18:47Z); the
  `date`/`python3 -c` forms are denied.
- Evidence is limited to the single collected 10-asset dataset; no temporal or
  cross-universe extrapolation.

## Next

Task R-006: correct `research/backtest/regime_stability.py::momentum_signals` to the
lookback-return rule (weight = +1 iff log(closes[i]) - log(closes[i-lookback]) > 0) and
re-run the momentum evidence base through the full gate stack, treating the prior
momentum SUPPORTED verdict as provisional. Then R-007 for `mean_reversion_signals`.