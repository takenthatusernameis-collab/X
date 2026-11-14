# Activation — 2026-11-14 (campaign run 43582348811, agent 02)

Observed UTC timestamps:
- Activation start: 2026-11-14T10:01:36Z (session message time; worker_progress captured 10:01:32Z)
- Substantive checkpoint (cost grid + independent verification complete): 2026-11-14T10:09:09Z
- Activation finish: 2026-11-14T10:10:33Z

## Task (R-001, HIGHER_ORDER_OBJECTIVE)

Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the
repository's existing realistic cost model and matched null, on the existing
10-asset collected universe, and independently verify the result.

Primary question: does the short-horizon momentum edge survive conservative
transaction-cost stress on the collected real-data universe?

## CHANGED

- `research/checks/momentum_cost_sensitivity.py` (new): predeclared cost grid
  {zero, realistic, 3x-conservative}, lookbacks {3,5,10}, 10-asset universe,
  matched coin-flip null per cost level, a-priori classification rule, and a
  cost-model audit.
- `research/checks/verify_momentum_cost_sensitivity.py` (new): independent
  verifier with 4 gates.
- `state/check_artifacts/momentum_cost_sensitivity_results.json` (new artifact).
- `state/LEARNING_STATE.md`: new frontier row, updated Next, new recent-history row.
- `state/worker_progress.md`: completion milestone.

No engine/framework/control-plane files changed. No new dependency.

## VERIFIED

- `python research/checks/momentum_cost_sensitivity.py`: manifest 10/10 OK,
  preflight PASSED, leakage + cost-accounting PASS (AMZN/JPM), full grid run,
  determinism AMZN realistic lookback=5 r1==r2.
- `python research/checks/verify_momentum_cost_sensitivity.py`: 4/4 gates MATCH.
  - Gate 1: fresh `walk_forward` + `noise_benchmark` (separate path, never
    `stress_segments`) for AMZN/JPM across all 3 cost levels x 3 lookbacks.
  - Gate 2: zero-cost column reproduces `momentum_lookback_sweep_results.json`
    on all 10 tickers x 3 lookbacks.
  - Gate 3: classification + determinism reproduce.
  - Gate 4: cost-model audit cross-check MATCH.
- `python -m unittest discover -s tests -q`: 168 tests, OK.

## Result (predeclared classification)

Edge counts (REGIME_STABLE & every segment median > 0), lookbacks 3/5/10:
- zero: 6 / 6 / 5
- realistic: 6 / 6 / 6
- conservative (3x): 6 / 7 / 7
=> OVERALL COST_ROBUST.

Candidate per-fold medians are essentially unchanged by cost: zero->conservative
decay 0.000 for AAPL/AMZN/GOOGL/JPM/META, 0.001 for JNJ/MSFT/XOM, 0.024 for NVDA
(a non-edge). The rise in edge counts at conservative cost is a null-dispersion
artifact (JPM REGIME_DEPENDENT->REGIME_STABLE because the cost-loaded coin-flip
null's cross-segment dispersion grows), not an improvement in the candidate.

## Cost-model audit (decision-relevant qualification)

The engine applies `close + slippage_cents/100 + close*slippage_proportional`
as the fill price on every fill. On a buy this is a cost; on a sell the seller
receives the higher price, so it is a credit. Over a round trip the two
approximately cancel. Round-trip probe: zero 909090.9091, slippage-only
909545.4545 (credit), commission-only 909086.9091 (cost). AMZN realistic
full-sample: 4451 fills, commissions $21,055.22, net slippage cash +$2,189.11
(credit), equity zero $6,537,258 vs realistic $6,518,392 -> drag 1.8866% of
capital over 4465 bars (~0.11%/yr), dominated by commissions.

Therefore COST_ROBUST is established for the commission channel of the existing
cost model only. Robustness to true directional slippage / market impact is
UNVERIFIED until the engine's slippage sign is repaired and this check is rerun.

## UNVERIFIED

- True directional-slippage / market-impact robustness (see above).
- The lookback 3/5/10 grid is near-degenerate: candidate medians are identical
  across lookbacks because sign(mean log return over the window) =
  sign(closes[i]/closes[i-lookback]-1); the grid does not independently stress
  lookback.
- Verifier defect found and repaired in-session: the first verifier run reported
  Gate-1 MISMATCH because it compared the last segment's fold count against the
  artifact's first-segment `n_folds`; fixed and re-run to MATCH. This is a
  verifier repair, not a research result.

## NEXT (one bounded action)

Repair the engine's directional slippage sign (charge slippage on sells instead
of crediting) and rerun `research/checks/momentum_cost_sensitivity.py` to test
true execution-cost robustness. Recorded as candidate task R-006.