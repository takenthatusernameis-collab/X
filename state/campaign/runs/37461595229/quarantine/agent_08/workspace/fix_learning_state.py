from pathlib import Path

p = Path('/home/runner/work/X/X/state/LEARNING_STATE.md')
text = p.read_text()

start_marker = "| Momentum (long the previous 5-day return, hold 1 day, daily rebalance) on collected data |\n"
end_marker = "| None -- cell closed TESTED; momentum admission now qualified with the horizon finding. |\n"
assert start_marker in text, "momentum header not found"
start = text.index(start_marker)
end = text.index(end_marker) + len(end_marker)

old_block = text[start:end]
print("OLD BLOCK LENGTH:", len(old_block))

new_block = ("| Momentum (long the previous N-day return, hold 1 day, daily rebalance) on collected data | "
    "| FALSIFIED (post-repair verification, agent 08, R-002, 2026-10-06) | "
    "Infrastructure defect: `momentum_signals` / `mean_reversion_signals` in "
    "`research/backtest/regime_stability.py` computed the mean of *log prices* `np.mean(np.log(closes[...]))` "
    "instead of the lookback-day log return `np.log(closes[i]) - np.log(closes[i - lookback])`; for any "
    "appreciably trending series (all 10 collected tickers) this made momentum permanently long the market "
    "(weight +1 everywhere) and reversal permanently short (weight -1 everywhere). The momentum "
    "\"7/10 uniformly-positive REGIME_STABLE edge\" and the reversal \"systematic losses everywhere\" verdict "
    "were both artifacts of this construction error. On the **corrected** signal: lookback-5 universe = "
    "3/10 REGIME_STABLE (all with mixed-sign medians only; e.g. AAPL [+0.085,+0.035,-0.037,+0.054], "
    "META [-0.129,-0.041,+0.072], NVDA [-0.024,+0.092,+0.063]), 6/10 CONSISTENT_WITH_NOISE, 1/10 "
    "REGIME_DEPENDENT; synthetic perturbation baseline +0.013 within sample-calibrated null tolerance - no "
    "uniformly-positive REGIME_STABLE edge. Hold-period sensitivity (R-002, grid hold=(1,2,3,5)): "
    "**HORIZON_DESTROYED** - 0 positive edges at any hold, all 10 assets NO_EDGE; artifact written to "
    "`state/check_artifacts/momentum_hold_period_sensitivity_results.json`; independent recomputation MATCH "
    "across all sections; determinism r1==r2; leakage review PASS on AMZN/JPM at every hold; perturbation "
    "medians [0.013, -0.001, -0.009, 0.001] within null tolerance at every hold. Lookback sweep "
    "(3/5/10): ROBUST=0, SENSITIVE=0, NO_EDGE=10. Longer-horizon (20/60): ROBUST=0, SENSITIVE=5, NO_EDGE=5. "
    "Conclusion: the prior admission of momentum as candidate positive evidence was computed from a defective "
    "signal and is withdrawn; on the corrected signal the momentum class shows no robust positive edge at the "
    "lookback, horizon, or holding-period gates. | None - cell closed FALSIFIED (post-repair). See agent_08 "
    "record.\n"
    "| Momentum: longer-horizon generalization (lookback 20 / 60) | FALSIFIED (post-repair, agent 08, "
    "R-002) | The longer-horizon run (37394344201) used the same defective signal. On the corrected signal: "
    "lookback-20 REGIME_STABLE=4 / lookback-60 REGIME_STABLE=3 (all mixed-sign medians); horizon-robustness "
    "ROBUST=0 (SENSITIVE=5, NO_EDGE=5). The \"edge generalizes to the 60-day horizon in magnitude\" claim was "
    "computed from a permanently-long signal and is withdrawn; no robust edge at either longer horizon. | None "
    "- cell closed FALSIFIED (post-repair); independent verifier `verify_momentum_longer_horizon.py` is stale "
    "and queued for repair (P-004).\n")

text = text[:start] + new_block + text[end:]
p.write_text(text)
print("Replaced momentum cell block. New block length:", len(new_block))

# Sanity: show the updated rows
lines = text.splitlines()
for i, line in enumerate(lines, 1):
    if 'momentum' in line.lower() or 'longer-horizon' in line.lower() or 'HORIZON_DESTROYED' in line:
        print(f"{i}: {line[:160]}")
