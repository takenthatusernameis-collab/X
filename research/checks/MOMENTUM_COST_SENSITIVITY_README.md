"""Momentum Cost-Sensitivity Test (R-001)

This document describes the bounded cost-sensitivity check for lookback 3/5/10
momentum under the repository's existing realistic cost model and matched null.

Task ID: R-001
Role: HIGHER_ORDER_OBJECTIVE

Primary Question:
Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe?

Scope:
Use the existing 10-asset collected universe and existing momentum implementation/check framework.

Objective:
Run a bounded cost-sensitivity check for lookback 3/5/10 momentum under the repository's existing realistic cost model and matched null.

Out of Scope:
Do not tune signal rules after seeing cost results, expand the universe, or introduce leverage.

Realistic Cost Model:
The test uses the realistic cost parameters established in the repository examples:
- commission_per_trade: $2.0 per trade
- commission_per_share: $0.003 per share
- slippage_cents: $0.02 per share (fixed)
- slippage_proportional: 5 bps (0.0005)

Methodology:
1. Load the collected 10-asset Yahoo Finance universe (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, JNJ, XOM)
2. For each lookback period (3, 5, 10 days):
   a. Generate momentum signals: long previous N-day return, hold 1 day, daily rebalance
   b. Run walk-forward IS/OOS validation with zero costs (baseline)
   c. Run walk-forward IS/OOS validation with realistic costs
   d. Run matched null: coin-flip random signals with realistic costs
   e. Compare realistic costs results vs matched null
3. Determine if momentum edge survives realistic costs (real_cost_median > null_median + tolerance)

Deliverable:
- momentum_cost_sensitivity_results.json artifact in state/check_artifacts/
- Contains per-asset and aggregate results for all three lookback periods
- Contains edge survival analysis and decision-relevant classification

Verification:
- Independent verification via verify_momentum_cost_sensitivity.py
- Matches existing repository patterns for cost sensitivity testing
- Uses same data, seeds, and methodology as other repository tests

Results Interpretation:
- If edge_survives_realistic_costs = true: momentum edge survives conservative transaction costs
- If edge_survives_realistic_costs = false: momentum edge does not survive conservative transaction costs
- Results are reproducible and follow repository evidence standards

Research Impact:
This addresses the bottleneck identified in the repository: "The candidate positive momentum evidence is short-horizon and mechanical; cost robustness remains decision-relevant."

Evidence Basis:
The existing momentum implementation admits short-horizon momentum as candidate positive evidence. This test provides the required cost robustness qualification for decision-making.

Constraints:
- Fixed lookback periods (3/5/10) - no tuning after results
- Fixed cost model - matches repository examples
- Fixed universe - existing 10-asset collected universe
- No leverage - position sizing follows existing BacktestConfig defaults
- No scope expansion - strictly bounded to cost sensitivity only

Next Steps:
- Analyze results for decision-making
- Update research state with findings
- Determine if additional robustness testing is needed
- Integrate results into evidence base for future research decisions

Note:
Research/simulation only. No live trading or production execution. This test is for research purposes only and must not be interpreted as trading evidence.