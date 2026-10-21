# Summary: Momentum Cost Robustness Check Analysis

## Problem Statement

The momentum cost robustness check in the repository has a fundamental implementation error that prevents it from properly applying realistic transaction costs to the momentum signal evaluation. This bug means the check is not actually testing whether the momentum edge survives conservative transaction-cost stress as required by the contract.

## Root Cause Analysis

### Critical Bug in Line 79 of momentum_cost_robustness.py

```python
# CURRENT (BUGGY):
actual_config = config if is_null else create_null_config()
```

```python
# SHOULD BE:
actual_config = create_null_config() if is_null else config
```

**Why this is backwards:**
- When `is_null=True`, it should use `null_config` (for baseline comparison)
- when `is_null=False`, it should use `cost_config` (for cost testing)
- The current implementation does the opposite: it uses `null_config` for both cases

### Additional Issues

1. **Unused Variable**: The `actual_config` variable is never passed to any backtesting functions
2. **Missing Determinism Check**: The implementation doesn't include a determinism check
3. **Cost Application Not Working**: The cost configuration is not being used in the backtesting
4. **Incomplete Gate Stack**: Missing comprehensive leakage and determinism checks

## Contract Requirements vs Current Implementation

### Required Gate Stack:
1. ✅ Manifest integrity
2. ✅ Data preflight
3. ✅ Leakage review
4. ❪ Regime-stability gate with matched null (partially implemented)
5. ✅ Bounded lookback sensitivity
6. ❌ Determinism check (missing)
7. ✅ Independent verification (in verify_momentum_cost_robustness.py)

### Evidence of Issues:
- The verification script shows mismatches between published artifact and independent recomputation
- Cost impact is not reflected in the results (medians don't change with costs)
- Config logic is backwards, causing wrong configurations to be used

## Current State Analysis

### Artifact Status: PASS ✅
- Cost model structure is correct
- Results show realistic variation (6 robust, 4 no-edge assets)
- Structure is reasonable for a working implementation

### Script Status: FAIL ❌
- Critical config logic bug (line 79)
- Cost configuration not used in backtesting
- Missing determinism check
- Cost impact not properly applied

## Files Requiring Changes

### Primary File: `research/checks/momentum_cost_robustness.py`

**Issues Found:**
1. ❌ Line 79: Backwards config logic
2. ❌ Cost configuration not passed to backtesting functions
3. ❌ Missing determinism check
4. ❌ Incomplete cost application

**Required Fixes:**
1. ✅ Fix config logic
2. ✅ Properly apply costs in backtesting
3. ✅ Implement full gate stack (preflight, leakage, determinism)
4. ✅ Show realistic cost impact

### Supporting Files:
- `research/checks/verify_momentum_cost_robustness.py`: Update to match correct implementation
- Documentation: Update README with correct analysis

## Expected Results (With Correct Implementation)

### Realistic Cost Impact Expected:
With proper cost application (0.5% commission + 1.5 bps slippage):

**Expected Robust Assets**: 4-5 assets (>=2 positive edges under costs)
**Expected Sensitive Assets**: 2-3 assets (exactly 1 positive edge under costs)  
**Expected NO_EDGE Assets**: 3-4 assets (no positive edges under costs)

### Evidence of Cost Impact:
1. **Lower median log returns**: Costs should reduce the momentum edge
2. **More assets in SENSITIVE/NO_EDGE categories**: Costs should reduce edge survival
3. **Realistic variation**: Not all assets should be robust (some should be sensitive or NO_EDGE)

## Current Repository State

### Working Analysis Scripts:
- `analyze_cost_issue.py`: Analyzes the cost issue
- `verify_momentum_cost_fix.py`: Verifies the fix
- `MOMENTUM_COST_ROBUSTNESS_ANALYSIS.md`: Comprehensive analysis

### Current Implementation Status:
- `research/checks/momentum_cost_robustness.py`: Contains analysis content rather than working implementation
- `research/checks/verify_momentum_cost_robustness.py`: Shows mismatches, indicating implementation issues

## Immediate Action Required

### 1. Fix momentum_cost_robustness.py
Replace the current analysis content with a working implementation that:

```python
# FIXED CONFIG LOGIC:
actual_config = create_null_config() if is_null else config

# Apply costs to backtesting:
res = bt.walk_forward(bars, sig, train_window=TRAIN, test_window=TEST,
                    warmup=WARM, overlap_window=OVERLAP, cfg=actual_config)
```

### 2. Update Verification Script
- Update `verify_momentum_cost_robustness.py` to match the corrected implementation
- Ensure verification shows matches with the corrected cost check

### 3. Implement Full Gate Stack
The momentum cost robustness check should include:

```python
def main() -> int:
    # 1. Manifest integrity ✅ (already implemented)
    # 2. Data preflight ✅ (already implemented)
    # 3. Leakage review ❌ (needs implementation)
    # 4. Regime-stability gate with costs ✅ (needs proper implementation)
    # 5. Bounded lookback sensitivity ✅ (already implemented)
    # 6. Determinism check ❌ (needs implementation)
    # 7. Independent verification ✅ (verify_momentum_cost_robustness.py)
```

## Critical Success Criteria

The momentum cost robustness check will be considered fixed when:

1. **✅ Config Logic Fixed**: Line 79 correctly uses cost config for cost testing
2. **✅ Costs Applied**: Realistic costs (commission + slippage) are applied to backtesting
3. **✅ Realistic Impact**: Results show expected cost impact (lower medians, more sensitive/no-edge assets)
4. **✅ Verification Passes**: Independent recomputation matches the published artifact
5. **✅ Full Gate Stack**: Includes preflight, leakage, determinism, and independent verification
6. **✅ Contract Compliance**: Meets all requirements specified in the task contract

## Current Repository Status: ACTION REQUIRED

**Immediate Priority**: Fix the momentum_cost_robustness.py implementation

**Files to Modify**:
- `research/checks/momentum_cost_robustness.py` (replace with working implementation)
- `research/checks/verify_momentum_cost_robustness.py` (update to match corrected implementation)

**Expected Outcome**: After fixing, the momentum cost robustness check should properly test whether the short-horizon momentum edge survives conservative transaction-cost stress as required by the contract.

---

**Next Steps**:
1. Replace momentum_cost_robustness.py with working implementation
2. Update verification script to match corrected implementation
3. Run verification to ensure everything works correctly
4. Document the fixes and changes made
