# Cost-Sensitive Momentum Check

This directory contains the cost-sensitive momentum robustness check for lookback 3/5/10 momentum on the collected real-data universe.

## Files

### research/checks/momentum_cost_sensitivity.py
The main cost-sensitive momentum check. Tests whether the short-horizon momentum edge (lookback 3 / 5 / 10) survives a realistic transaction-cost model on the collected real-data universe.

**Key Features:**
- Uses realistic costs: commission_per_trade=$2.00, commission_per_share=0.3%, slippage=$0.02 per share, slippage_proportional=5bps
- Walk-forward validation with 4 contiguous volatility blocks
- Train=252d / test=84d / warmup=60d / overlap=60d, min 400 bars/segment
- Matched coin-flip null comparison
- Creates artifact: `state/check_artifacts/momentum_cost_sensitivity_results.json`

### research/checks/verify_momentum_cost_sensitivity.py
Independent verifier for the cost-sensitive momentum check.

**Verification Gates:**
1. Cross-consistency check with existing momentum_results.json
2. Leakage review per asset
3. Universe-level verification
4. Reproducibility gate for robustness verdicts

### research/checks/test_momentum_cost_sensitivity.py
Unit tests for the cost-sensitive momentum check.

**Tests:**
- BacktestConfig configuration with realistic costs
- momentum_signals function validation
- File creation verification

## Cost Model

The realistic cost model matches the one used in `examples/ma_crossover.py`:

```python
COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    commission_per_trade=2.0,      # $2.00 per round trip
    commission_per_share=0.003,    # 0.3% per share
    slippage_cents=2.0,            # $0.02 per share
    slippage_proportional=0.0005, # 5 bps
    warmup_periods=WARM,
)
```

## How to Run

1. Run the main check:
```bash
python research/checks/momentum_cost_sensitivity.py
```

2. Run the verifier:
```bash
python research/checks/verify_momentum_cost_sensitivity.py
```

3. Run unit tests:
```bash
python -m pytest research/checks/test_momentum_cost_sensitivity.py -v
```

## Expected Output

The check will output:
- Manifest integrity verification
- Data preflight checks
- Leakage review for AMZN/JPM
- Cost-sensitive momentum lookback sweep results
- Determinism verification
- Artifact file written to `state/check_artifacts/momentum_cost_sensitivity_results.json`

## Artifact

The artifact contains:
- Dataset information
- Lookback parameter settings
- Per-asset results for lookbacks 3, 5, 10
- Cost configuration used
- Robustness verdicts
- Total costs incurred

## Verification

The verifier will check:
1. Cross-consistency with existing momentum_results.json
2. Data integrity and leakage checks
3. Correctness of results
4. Reproducibility of robustness verdicts

## Integration

This check is designed to be integrated into the existing research framework:
- Follows the same patterns as other checks (momentum_lookback_sweep, momentum, etc.)
- Uses the same validation gates (preflight, leakage, determinism)
- Produces the same artifact format
- Has independent verification

## Purpose

This check addresses the bottleneck identified in the agent contract:
"The candidate positive momentum evidence is short-horizon and mechanical; cost robustness remains decision-relevant."

It tests whether the short-horizon momentum edge survives conservative transaction-cost stress on the collected real-data universe, which is crucial for determining whether the momentum signal should be admitted to the evidence base.