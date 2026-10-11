# Cost Sensitivity Analysis for Lookback 3/5/10 Momentum

## Objective
To answer the primary research question: **Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe?**

## Methodology
This analysis runs momentum signals with realistic transaction costs (from `tests/test_engine.py`) and compares the edge against a matched null with the same costs.

### Realistic Cost Model
Based on the repository's existing realistic cost model from `tests/test_engine.py`:
- `commission_per_share`: 0.003 (3 basis points per share)
- `slippage_cents`: 5.0 (5 cents per share)  
- `slippage_proportional`: 0.001 (10 bps proportional slippage)

### Process
1. **Run momentum lookback sweep (3, 5, 10)** using the existing `momentum_lookback_sweep.py` pattern
2. **Apply realistic costs** to both momentum signals and matched null benchmarks
3. **Compare results** with no-cost baseline to assess cost sensitivity
4. **Generate artifact** showing how realistic transaction costs affect the momentum edge

### Implementation Files
- `research/checks/momentum_cost_sensitivity_simple.py` - Primary cost-sensitivity check
- `research/checks/verify_momentum_cost_sensitivity_simple.py` - Independent verification
- `research/checks/momentum_cost_sensitivity.py` - Original comprehensive attempt
- `research/checks/verify_momentum_cost_sensitivity.py` - Verification for comprehensive version

## Key Findings (to be populated after execution)
The analysis will reveal:

1. **Impact of realistic costs on momentum edge**
   - How much the positive momentum edge diminishes under realistic transaction costs
   - Which lookback horizons are most/least cost-sensitive

2. **Cost-robustness verdicts**
   - ROBUST: Edge survives realistic costs at >= 2 lookbacks
   - SENSITIVE: Edge only at exactly 1 lookback under costs  
   - NO_EDGE: No edge under realistic costs

3. **Comparison with no-cost baseline**
   - Quantify the cost drag on momentum performance
   - Assess whether momentum remains viable after transaction costs

## Expected Output
- `state/check_artifacts/momentum_cost_robustness_results.json` - Main artifact with cost-sensitivity analysis
- Deterministic, reproducible results that can be independently verified
- Clear classification of cost robustness for the momentum signal class

## Research Significance
This analysis addresses the bottleneck identified in the agent contract: "The candidate positive momentum evidence is short-horizon and mechanical; cost robustness remains decision-relevant."

By testing momentum under realistic costs, we can determine whether:
1. The observed momentum edge is robust enough to withstand real-world trading costs
2. Momentum should be admitted to the evidence base as candidate positive evidence
3. Short-horizon momentum strategies remain viable after accounting for transaction costs

## Verification
Both the primary check and independent verification follow the established patterns from other checks in the repository (e.g., `verify_momentum_lookback_sweep.py`).
