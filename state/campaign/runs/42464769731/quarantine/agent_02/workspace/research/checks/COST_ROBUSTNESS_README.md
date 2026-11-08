# Momentum Cost Robustness Analysis

## Overview
This repository contains code to test whether the short-horizon momentum edge 
(long previous N-day return, hold 1 day, daily rebalance) survives realistic 
trading costs on the collected real-data universe.

## Files Created

### 1. research/checks/momentum_cost_robustness.py
**Objective**: Run a bounded cost-sensitivity check for lookback 3/5/10 momentum 
under the repository's existing realistic cost model and matched null.

**Method**:
- Per-asset walk-forward (train=252d, test=84d, warmup=60d, overlap=60d) 
- Cost model: commission_per_trade=2.0, commission_per_share=0.003, 
  slippage_cents=2.0, slippage_proportional=0.0005
- Matched coin-flip null benchmark with same costs
- Cost robustness classification: CONSISTENT_WITH_COSTS if candidate edge > null edge + tolerance

**Output**: 
- Artifact: state/check_artifacts/momentum_cost_robustness_results.json
- Contains per-asset cost robustness results for lookbacks 3, 5, 10

### 2. research/checks/verify_momentum_cost_robustness.py
**Objective**: Independent verification of momentum_cost_robustness.py

**Design**:
- Re-implementation of key functions (volatility_blocks, momentum_signals)
- Independent walk_forward validation
- Artifact comparison to detect defects
- Follows same verification pattern as verify_momentum.py

### 3. state/campaign/runs/42464769731/agents/agent_02.json
**Objective**: Higher-Order Objective R-001
**Evidence**: Created script ready for execution
**Next**: Execute momentum_cost_robustness.py
**Decision**: REJECT (script created, ready for execution)

## How to Run

### Execute the Cost Robustness Check
```bash
# Navigate to repository root
cd /home/runner/work/X/X

# Run the momentum cost robustness check
python3 research/checks/momentum_cost_robustness.py
```

### Run Independent Verification
```bash
# Navigate to repository root
cd /home/runner/work/X/X

# Run independent verification
python3 research/checks/verify_momentum_cost_robustness.py
```

## Evidence Basis

From research/LEARNING_STATE.md:
- Momentum (lookback=5) shows 7/10 assets with REGIME_STABLE positive edge
- Lookback-sweep verification found 6/10 assets lookback-robust (positive at >=2 lookbacks)
- Cross-sectional momentum test FALSIFIED (consistent with noise)

**Question**: Does the short-horizon momentum edge survive conservative 
transaction-cost stress on the collected real-data universe?

**Gap**: Cost robustness remains decision-relevant despite positive evidence

## Success Criteria

1. **Reproducible**: Same inputs should produce identical outputs
2. **Independent verification**: Separate implementation confirms results
3. **Deterministic**: Artifact comparison shows no discrepancies
4. **Scope adherence**: No tuning, universe expansion, or leverage introduction

## Expected Output

### momentum_cost_robustness_results.json
- Dataset information
- Lookbacks: [3, 5, 10]
- Cost type: realistic
- Per-asset cost robustness results
- Summary: CONSISTENT_WITH_COSTS / INSUFFICIENT_COST_ROBUSTNESS counts

### Verification Output
- Manifest integrity check
- Artifact loading confirmation
- Per-asset verification results
- Summary verification status

## Integration with Research Frontier

**Current Status**: Momentum boundary cells tested and closed:
- lookback 5 momentum: ADMITTED AS CANDIDATE POSITIVE EVIDENCE
- lookback-sweep verification: lookback-robustness qualified
- cross-sectional momentum: FALSIFIED (noise-like)
- breakout filter: FALSIFIED (adds no robust evidence beyond momentum)

**Open qualification**: Conservative transaction-cost sensitivity test
- Fixed cost model, lookback 3/5/10, matched null
- Real-data universe (10 collected tickers)
- Result will inform whether momentum should be admitted to evidence base

## Files Status

✅ **Created**: 
- momentum_cost_robustness.py
- verify_momentum_cost_robustness.py  
- agent_02.json

⏳ **Ready for execution**: 
- momentum_cost_robustness.py (main task)
- verify_momentum_cost_robustness.py (verification)

## Research Impact

This cost robustness test addresses the critical gap in the momentum research:
1. **Validation**: Tests whether mechanical edge survives realistic costs
2. **Qualification**: Provides go/no-go for momentum admission to evidence base
3. **Independence**: Separate verification ensures no implementation defects
4. **Decision support**: Informs next research steps based on cost sensitivity

## Notes

- All scripts use repository-native imports (research.backtest)
- Cost model matches examples/ma_crossover_real_data.py
- Follow existing patterns from momentum_lookback_sweep.py
- Artifact format consistent with other check results
- Verification uses independent code paths for defect detection

## Next Steps

1. **Execute momentum_cost_robustness.py** to generate cost robustness results
2. **Run verify_momentum_cost_robustness.py** for independent verification
3. **Analyze results** to determine momentum cost robustness status
4. **Update research frontier** based on cost robustness findings
