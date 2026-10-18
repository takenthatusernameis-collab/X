"""Missing: momentum_volatility_normalized_signals function in research/backtest/regime_stability.py.
This is the core bottleneck preventing frontier-first selection from completing.

Agent 8 implemented the function but it was not persisted to the main codebase.
The momentum_volatility_normalized.py check cannot run because the function
it imports (bt.momentum_volatility_normalized_signals) does not exist.

This single missing function blocks the frontier-first selection process:
- Momentum was qualified as candidate positive evidence
- Volatility-normalized sizing framework was supposed to be implemented to test generalization
- But the framework was never successfully integrated
- Therefore, the momentum boundary review cannot be completed

Agent 9's task is to audit this activation history and identify this incomplete frontier-first selection, then complete it by implementing the missing function.

The solution is straightforward:
1. Add momentum_volatility_normalized_signals function to research/backtest/regime_stability.py
2. Export it from research/backtest/__init__.py
3. Migrate all quarantine files to main codebase
4. Run momentum_volatility_normalized.py to complete the evaluation

This will restore the complete frontier-first selection process and demonstrate its value.
"