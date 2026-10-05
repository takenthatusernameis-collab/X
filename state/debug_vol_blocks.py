# Superseded — activation 37263019969 (2026-10-05)
#
# Diagnostic written to investigate apparent segment-label differences between
# research/backtest.volatility_blocks and the verifier's re-implementation.
# Result: the label arrays are IDENTICAL for all tickers. The verifier's earlier
# "MISMATCH" reports for AAPL/MSFT/JNJ/XOM were caused by a stale `lbls` variable
# (referencing the per-asset loop's AMZN labels) in the verifier's universe loop,
# not by label differences.
#
# The verifier was repaired to compute lbls = [lab for lab, _, _ in runs] inside
# the universe loop (see
# research/checks/verify_mean_reversion.py, universe-level recomputation section).
# After the repair, all universe medians and verdicts MATCH the check artifact.
#
# This file is superseded by that repair; it is retained only as a record of the
# investigation.
