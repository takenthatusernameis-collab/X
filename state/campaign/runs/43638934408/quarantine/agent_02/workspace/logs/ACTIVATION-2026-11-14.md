# Activation Log — 2026-11-14

## Activation — run 43638934408, campaign slot 02, agent 02 (task R-001: momentum cost-sensitivity)

**Activation start (UTC):** 2026-11-14T20:53:39Z (controller `agent_02_before.json` captured_utc; observed).
**Activation finish (UTC):** 2026-11-14T21:05:26Z (observed via runner clock in-session).
**Environment observation:** GitHub Actions Linux runner executing a fresh Kilo CLI process; repository root `/home/runner/work/X/X`, ref `main`, workflow `Kilo Research Sequential Learning Campaign`. Infrastructure only — not research evidence.

**Objective (task R-001):** Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null; produce a reproducible artifact and an independent verification result classifying cost robustness.

**Primary question:** Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe?

**Selection basis:** focused-task contract (highest evidence-backed qualitative priority among OPEN HIGHER_ORDER_OBJECTIVE tasks; R-001 selected over R-002/R-003/R-005).

**A-priori falsification prediction (declared before any cost result was observed):**
1. Candidate absolute net medians decline monotonically across zero -> realistic -> stress2x (daily rebalancing pays cost on every flip).
2. The matched coin-flip null declines too, and by more than the candidate, because the iid {-1,0,+1} null changes position on ~2/3 of bars while momentum changes less often; so the candidate-minus-null gap may widen even as absolute returns fall.
3. Primary classification at the realistic level: the zero-cost positive-edge assets (AAPL, MSFT, GOOGL, AMZN, META, TSLA) retain a positive REGIME_STABLE edge (SURVIVES); the others stay NO_EDGE. Falsifiable form: NO asset with a zero-cost positive edge is reclassified ERODED at the realistic level.

## Work performed (chronological)

1. **Read durable state:** `state/LEARNING_STATE.md`, `state/STATE.md` (momentum sections), `state/campaign/task_queue.json`, `state/campaign/current_task.json`, `agent_02_before.json`; read the existing momentum check (`research/checks/momentum_lookback_sweep.py`), engine (`research/backtest/engine.py`), perturbation/null (`research/backtest/perturbation.py`), and the repository's realistic cost model in `examples/ma_crossover_real_data.py` / `examples/volatility_regime_filter.py`.
2. **Baseline regression:** `python -m unittest discover -s tests -q` -> 168 tests OK before changes.
3. **New check:** `research/checks/momentum_cost_sensitivity.py` — manifest 10/10 OK; preflight PASSED; leakage PASS on AMZN/JPM with the realistic-cost fill-equity audit; cost grid zero / realistic / stress2x; walk-forward per segment train=252/test=84/warmup=60/overlap=60; 4 volatility blocks; min 400 bars/segment; each (lookback, cost level) candidate vs a matched coin-flip null under the SAME cfg; zero-cost column cross-consistency vs `momentum_lookback_sweep_results.json`; determinism r1==r2; artifact `state/check_artifacts/momentum_cost_sensitivity_results.json`.
4. **Signal-validity gate added (after an implausibly small cost drag was observed):** recomputes the true previous-N-day return sign as `log(close[i]) - log(close[i-lookback])` and requires >= 95% sign agreement. This gate FAILS for all 10 assets.
5. **Independent verifier:** `research/checks/verify_momentum_cost_sensitivity.py` — 5 gates (fresh cost recomputation on AMZN/JPM; independent signal-validity recomputation; zero-cost cross-consistency; classification reproducibility; determinism) all MATCH.
6. **Unit tests:** `tests/test_momentum_cost.py` (12 tests: cost-config provenance, signal validity, positive-edge rule, SURVIVES/ERODED/NO_EDGE). Regression suite: 180/180 pass (168 baseline + 12 new).

## Results

- **Decisive finding — the tested "momentum" signal is not momentum.** `research/backtest/regime_stability.py::momentum_signals` (and `mean_reversion_signals`) compute `np.mean(np.log(closes[i-lookback+1:i+1]))` — the MEAN LOG PRICE over the window — not the previous-N-day return `log(close[i]) - log(close[i-lookback])`. For every collected ticker (prices > $1) the mean log price is positive on ~every bar, so the momentum weight is +1 on 4460/4465 bars with ONE position flip (dominant weight share 1.000). The signal's sign agrees with the TRUE previous-5-day return sign on only 0.538–0.580 of bars (near coin-flip). NVDA is 0.585 dominant / 2 flips only because its split-adjusted price dipped below $1.
- **Primary question verdict: `UNVERIFIED_INVALID_SUBJECT`** — the cost grid is evaluated and verified, but its subject is a near-constant long (directional beta), not momentum.
- **Cost mechanics:** candidate cost drag ~0 (constant position, minimal signal-driven turnover); the iid {-1,0,+1} null turns over ~2/3 of bars and pays more cost, so its median falls and the candidate-minus-null gap WIDENS under cost (AMZN +0.078/+0.085/+0.092 at zero/realistic/2x). Signature of a directional-beta artifact, not a cost-robust momentum edge.
- **Mechanical cost classification (reported for completeness, NOT interpretable as momentum cost robustness):** realistic SURVIVES=5 (AAPL, GOOGL, AMZN, META, TSLA), ERODED=1 (MSFT), NO_EDGE=4 (NVDA, JPM, JNJ, XOM); stress2x SURVIVES=4, ERODED=2 (+TSLA), NO_EDGE=4. Per-lookback positive-edge counts (lookback 3/5/10): zero 6/6/5, realistic 6/6/6, stress2x 5/5/6.
- **Invalidated prior evidence (same defect):** reversal FALSIFIED (37263008074, constant short) and the momentum admission/qualifications (37318950814, 37361000967, 37394344201, `momentum_horizon_gap_analysis`). The earlier "momentum vs reversal P&L correlation -0.99" corroboration is itself a symptom (always-long vs always-short must be ~-1). MA-crossover, breakout, cross-sectional relative-strength, and cross-sectional-momentum are unaffected (correct return/rank math).

## Verdict

**VERIFIED_NEGATIVE_RESULT.** The cost question is not adjudicable because the tested signal is mis-implemented; the negative/invalidating finding (the momentum/reversal evidence chain rests on a mean-log-price bug, and the cost result is an artifact of a constant directional bet) is independently verified. **Process decision: IMPROVE** — the gate stack needs a signal-construction validity gate before a signal class is admitted; a near-constant directional bet passed five prior activations and their independent verifiers as "momentum."

## Change summary (what actually changed)

- **CHANGED:** `research/checks/momentum_cost_sensitivity.py` (new); `research/checks/verify_momentum_cost_sensitivity.py` (new); `tests/test_momentum_cost.py` (new); `state/check_artifacts/momentum_cost_sensitivity_results.json` (new artifact); `state/STATE.md` (new activation section); `state/LEARNING_STATE.md` (Next line superseded; momentum rows marked SUPERSEDED; new frontier row; new activation section); `state/worker_progress.md` (milestone); `logs/ACTIVATION-2026-11-14.md` (this log).
- **VERIFIED:** manifest 10/10; preflight PASSED; leakage PASS on AMZN/JPM (realistic-cost fill-equity audit); zero-cost cross-consistency vs prior sweep MATCH (all 10 x 3); independent verifier 5/5 gates MATCH; determinism identical; 180/180 regression tests pass.
- **UNVERIFIED:** controller-side `validate_campaign_agent.py` on this record (run by the controller after the session; not executed here); the corrected-momentum cost result (not run — the signal fix is the next bounded action); whether the corrected signal changes the momentum/reversal verdicts (untested).
- **NEXT (single bounded action):** fix `momentum_signals`/`mean_reversion_signals` to use the previous-N-day log return, add a sign-agreement unit test, and re-run the momentum evidence chain (lookback 3/5/10 through the cost grid) before any cost conclusion.

Research / simulation only. No live trading or production execution.