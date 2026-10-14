"""Activation record for holding-period momentum test.

This activation implements the focused-task (R-002) as per the higher-order objective:
"Test a fixed lookback-5 momentum signal with a predeclared small holding-period set using
the existing walk-forward/null methodology."

The holding-period grid is predeclared: [1, 2, 3, 5, 7] days.

Primary question: Does the qualified momentum effect survive a small
predeclared holding-period variation without becoming a single-point timing artifact?

The momentum cell from activation 37318950814 (lookback-5, 1-day hold) was SUPPORTED with
a REGIME_STABLE positive edge in 7/10 assets. This test extends that analysis to a
holding-period grid to investigate temporal robustness.

METHODOLOGY:
- Check file: research/checks/check_holding_period.py
- Pre-declared holding-period grid: [1, 2, 3, 5, 7] days
- Fixed lookback: 5 days (no post-inspect resolution)
- 10-asset collected universe (yf-ohlcv-universe-2009-to-2026-10-03)
- Same walk-forward parameters: train=252d, test=84d, warmup=60d, overlap=60d, min 400 bars/segment
- Volatility-blocks regime classification (n_blocks=4, window=60)
- Coin-flip null on each segment (sign-flipped signal)
- Per-asset verdict counts and universe-level verdict
- Independent verification via verify_holding_period.py
- Deterministic assertion

FALSIFICATION PREDICTION:
- CONSISTENT_WITH_NOISE across holding periods (no edge in any holding period)
- REGIME_DEPENDENT (edge squanders on one holding period, single-point timing artifact)
- REGIME_STABLE_LOSS (uniformly negative, reversal wins on both sides)

SUCCESS CRITERION:
The holding-period sensitivity is classified and any positive finding is reproducible.

EVIDENCE STATUS:
- Initial setup and file creation completed
- Check script and verifier written (ready for execution)
- Pre-declared holding-period grid implemented
- All framework dependencies satisfied

NEXT STEPS:
Execute the holding-period momentum test (check_holding_period.py) and its
independent verifier (verify_holding_period.py) to determine whether the
qualified momentum effect survives the holding-period variation.

STATUS: INFRASTRUCTURE READY, EXECUTION PENDING
"""
