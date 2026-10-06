# Worker Progress

activation_id: 37394344201
phase: DEEP
status: ACTIVE
last_verified_milestone: longer-horizon momentum cell executed and verified end-to-end. Path 1 (`momentum_longer_horizon.py`): lookback-20 REGIME_STABLE=6 / lookback-60 REGIME_STABLE=4; horizon-robustness ROBUST=3 (AAPL, AMZN, JPM), SENSITIVE=3 (GOOGL, META, TSLA), NO_EDGE=4; determinism r1==r2. Path 2 (`verify_momentum_longer_horizon.py`): Gate 1 fresh walk_forward recomputation AMZN/JPM MATCH; Gate 2 fresh medians/nulls for all 10 assets + regenerated verdicts MATCH; Gate 3 determinism identical. Path 3 (`momentum_horizon_gap_analysis.py`): independent gap recharacterization — candidate medians identical across lookback 20/60 for all 10 assets; positive candidate-vs-null gaps in all segments at both lookbacks for every asset (2/3 segments for NVDA); determinism identical. Key finding: the apparent 60-day verdict "degradation" is a coin-flip-null dispersion artifact, not an edge decay; the momentum edge is horizon-stable in magnitude. 134/134 regression tests pass.

next_bounded_action: record the findings in state/LEARNING_STATE.md (frontier), state/STATE.md, state/activation_status.json, and logs/ACTIVATION-2026-10-06.md, then close the activation.

This file is a liveness contract. Update it only after a real research-state transition or verified milestone.

## Active objective

Longer-horizon momentum on the collected universe: test whether the lookback=5 momentum edge (admitted as candidate positive evidence in activation 37318950814, verified lookback-3/5/10 robustness in activation 37361000967) survives at the classic momentum formation windows lookback 20 and 60.

Method (fixed a-priori): long the previous 20-/60-day return, hold 1 day, daily rebalance; walk-forward per segment train=252d/test=84d/warmup=60d/overlap=60d; 4 contiguous volatility blocks; each lookback vs a coin-flip sign null on the same segments; universe sweep across all 10 collected tickers; determinism asserted; artifact written to state/check_artifacts/momentum_longer_horizon_results.json; independent verifier recomputes the artifact via a fresh walk_forward path.

Expected effect: reduces uncertainty about whether the momentum edge is confined to the short-horizon window (3-10 days) or generalizes to the classic 1-3 month formation window; the result is admitted only if it passes the regime gate, the independent-verification gate, and determinism.
