# Momentum Cost Sensitivity Check

## Overview

The momentum cost sensitivity check is a research activation that tests whether the short-horizon momentum edge (supported in momentum.py) survives conservative transaction costs under the repository's existing realistic cost model and matched null.

## Objectives

1. **Test Momentum Lookback Robustness**: Evaluate whether the momentum edge with lookbacks 3, 5, and 10 survives realistic transaction costs
2. **Use Existing Realistic Cost Model**: Apply the repository's established realistic cost configuration:
   - `commission_per_trade`: 2.0
   - `commission_per_share`: 0.003
   - `slippage_cents`: 2.0
   - `slippage_proportional`: 0.0005
3. **Compare Zero-Cost vs Realistic-Cost**: Determine how momentum performance changes when realistic costs are applied
4. **Produce Reproducible Results**: Create an artifact and independent verification script

## Method

### Data Processing
- Dataset: Collected Yahoo Finance OHLCV universe (10 tickers, 2009-2026-10-03)
- Walk-forward validation: train=252d/test=84d/warmup=60d/overlap=60d
- Minimum segment bars: 400
- Volatility blocks: 4 contiguous blocks using past-only volatility classifier

### Two Configuration Modes
1. **Zero-cost baseline**: Uses `BacktestConfig(initial_capital=1e6, warmup_periods=60)`
2. **Realistic costs**: Uses the realistic cost model from `examples/ma_crossover_real_data.py`

### Signal Processing
- Momentum signal: long the previous N-day return, hold 1 day, daily rebalance
- Lookback values: 3, 5, 10
- Each lookback tested separately with the same framework as momentum_lookback_sweep_results.json

### Matched Null
- Coin-flip sign null benchmark for each segment
- Same null used for both zero-cost and realistic-cost comparisons
- Eliminates look-ahead bias and ensures fair testing

## Expected Outputs

### Momentum Cost Sensitivity Check (`research/checks/momentum_cost_sensitivity.py`)
1. **Manifest integrity**: Verify `research/data/manifest.json` checksums
2. **Data preflight**: Run `research/data/preflight.py` (data quality, preflight checks)
3. **Leakage review**: Signal integrity and equity-fill audit for AMZN/JPM (lookback=5)
4. **Load zero-cost results**: Read `momentum_lookback_sweep_results.json` (momentum_lookback_sweep_results.json)
5. **Run realistic cost lookback sweep**: Test momentum lookbacks 3/5/10 under realistic costs
6. **Cost-sensitivity analysis**: Compare zero-cost vs realistic-cost results
7. **Write artifact**: Create `momentum_cost_sensitivity_results.json` for independent verification

### Independent Verification (`research/checks/verify_momentum_cost_sensitivity.py`)
1. **Re-implement core functions**: Independent volatility_blocks, segment reconstruction, and momentum_signals
2. **Fresh recomputation**: Re-run walk_forward on all segments using the same inputs
3. **Artifact comparison**: Verify all medians and verdicts match the check's artifact
4. **Error reporting**: Report any mismatches and investigation requirements

## Research Significance

### Primary Question
"Does the short-horizon momentum edge survive conservative transaction-cost stress on the collected real-data universe?"

### Decision Context
Based on the existing evidence:
- Momentum is currently **ADMITTED AS CANDIDATE POSITIVE EVIDENCE** in the repository's LEARNING_STATE.md
- The edge is **short-horizon and mechanical** (noted as a caveat in current research)
- The existing edge is **3/10 robust** (ROBUST=6, SENSITIVE=0, NO_EDGE=4 in momentum_lookback_sweep_results.json)

### Research Gap
The current research has established momentum's existence under zero-cost conditions but **not its economic viability under realistic transaction costs**.

## Out-of-Scope
- **Do not tune signal rules**: After seeing cost results
- **Do not expand the universe**: Use only the existing 10-asset collected universe
- **Do not introduce leverage**: Use only the existing realistic cost model

## Expected Research Impact

### Positive Outcome Scenario
- Momentum edge survives realistic costs with minimal reduction
- Provides evidence that momentum is economically meaningful for trading
- Supports momentum's admission to the evidence base as robust

### Cost-Sensitive Scenario
- Momentum edge significantly eroded or eliminated by realistic costs
- Highlights transaction cost barrier to momentum profitability
- Informs research direction (e.g., focus on longer-horizon momentum)

### Research Integrity
- **Reproducible artifact**: Complete artifact with independent verification
- **Deterministic testing**: All results reproducible across independent runs
- **Transparent methodology**: Clear documentation of cost model and null hypothesis

## Verification Requirements

1. **Artifact Existence**: `state/check_artifacts/momentum_cost_sensitivity_results.json` must exist
2. **Independent Verification**: `research/checks/verify_momentum_cost_sensitivity.py` must run and report "All independent recomputations MATCH the check artifact."
3. **Regression Tests**: All existing tests must continue to pass
4. **Documentation**: Clear activation record in `logs/ACTIVATION-*.md` with UTC timestamp

## Success Criteria

- **Reproducible**: All results can be independently recomputed and verified
- **Economically Meaningful**: Clear evidence of momentum's viability under realistic costs
- **Methodologically Sound**: Proper use of matched null, realistic cost model, and existing framework
- **Well-Documented**: Clear research record with timestamps and evidence provenance