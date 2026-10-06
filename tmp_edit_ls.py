p = "/home/runner/work/X/X/state/LEARNING_STATE.md"
s = open(p).read()

# 1) Update the momentum frontier row's "Next:" cell -> closed status.
old_next = ("| Next: test generalization beyond the short-horizon window: "
            "run longer-horizon momentum (lookback 20/60) through the same "
            "regime + independent-verification gates before declaring momentum "
            "robust across horizons. |")
new_next = ("| Tested (37394344201): the momentum edge generalizes to lookback "
            "20 and 60 in magnitude; the 60-day regime-stability verdicts are "
            "reported as null-dispersion-sensitive rather than indicating an "
            "edge decay. Cell closed TESTED. |")
assert old_next in s, "momentum next cell not found"
s = s.replace(old_next, new_next)

# 2) Insert the longer-horizon frontier row right after the momentum row.
new_row = (
    "| Momentum: longer-horizon generalization (lookback 20 / 60) | "
    "TESTED (activation 37394344201, 2026-10-06) | "
    "Lookback-5 momentum (7/10 REGIME_STABLE) was admitted as candidate "
    "positive evidence. A longer-horizon sweep (lookback 20 and 60) was "
    "executed with three independent paths: (1) `momentum_longer_horizon.py` "
    "-- lookback-20 REGIME_STABLE=6 / lookback-60 REGIME_STABLE=4 across the "
    "10-asset universe; horizon-robustness ROBUST=3 (AAPL, AMZN, JPM), "
    "SENSITIVE=3 (GOOGL, META, TSLA), NO_EDGE=4 (MSFT, NVDA, JNJ, XOM); "
    "determinism r1==r2. (2) `verify_momentum_longer_horizon.py` -- Gate 1 "
    "fresh walk_forward + noise_benchmark recomputation of AMZN/JPM MATCH; "
    "Gate 2 fresh medians/nulls for all 10 assets + regenerated regime/horizon "
    "verdicts MATCH; Gate 3 determinism identical. (3) "
    "`momentum_horizon_gap_analysis.py` -- null-dispersion-robust "
    "recharacterization (independent fresh path): candidate per-segment "
    "medians are IDENTICAL across lookback 20 and 60 for all 10 assets; "
    "every asset keeps positive candidate-vs-null gaps in all segments at "
    "both lookbacks (2/3 segments for NVDA); determinism identical. Key "
    "finding: the apparent 60-day verdict \"degradation\" (4/10 REGIME_STABLE "
    "at 60 vs 6/10 at 20) is entirely a coin-flip-null dispersion artifact "
    "-- the null dispersion varies wildly with lookback (e.g. TSLA 0.603 -> "
    "0.013, GOOGL 0.034 -> 0.006, NVDA 0.022 -> 0.132) rather than an edge "
    "decay; the momentum edge is horizon-stable in magnitude. Decision: the "
    "momentum cell (lookback 5) remains ADMITTED AS CANDIDATE POSITIVE "
    "EVIDENCE, now qualified: the edge generalizes to the 60-day horizon in "
    "magnitude; the regime-stability verdict at 60-day is reported as "
    "null-dispersion-sensitive. | "
    "None -- cell closed TESTED; momentum admission now qualified with the "
    "horizon finding. |\n"
)
marker = ("| Momentum (long the previous 5-day return, hold 1 day, daily "
          "rebalance) on collected data |")
assert marker in s, "momentum row marker not found"
s = s.replace(marker, marker + "\n" + new_row)

# 3) Insert the learning-history row before the "## Learning rules" section.
new_hist_row = (
    "| 37394344201 (2026-10-06) | Execute the longer-horizon momentum frontier "
    "cell (lookback 20/60) through the same regime + independent-verification "
    "gates; then a null-dispersion-robust gap recharacterization | "
    "VERIFIED: (1) `momentum_longer_horizon.py` executed end-to-end (manifest "
    "10/10 OK, PREFLIGHT PASSED, leakage PASS on AMZN/JPM): lookback-20 "
    "verdict counts REGIME_STABLE=6 / lookback-60 REGIME_STABLE=4; "
    "horizon-robustness ROBUST=3, SENSITIVE=3, NO_EDGE=4; determinism r1==r2; "
    "artifact written. (2) `verify_momentum_longer_horizon.py` executed: Gate "
    "1 fresh walk_forward + noise_benchmark (AMZN/JPM lookbacks 20/60) MATCH; "
    "Gate 2 fresh medians/nulls for all 10 assets + regenerated regime/horizon "
    "verdicts MATCH; Gate 3 determinism identical. (3) "
    "`momentum_horizon_gap_analysis.py` executed: independent gap "
    "recharacterization -- candidate medians identical across lookbacks for "
    "all 10 assets, positive gaps in all segments at both horizons for every "
    "asset (2/3 for NVDA); determinism identical. 134/134 regression tests "
    "pass. Key finding: the momentum edge is horizon-stable in magnitude; the "
    "regime-stability verdicts at 60-day are null-dispersion-sensitive, not "
    "evidence of edge decay. | "
    "A positive candidate admission (lookback 5) is now qualified and "
    "strengthened by the horizon sweep: short-horizon momentum survives at "
    "the 60-day horizon in magnitude, and the framework's regime-stability "
    "verdict at longer lookbacks is reported with a null-dispersion caveat. "
    "Frontier-first selection operated correctly and closed the cell with a "
    "decisive, independently-verified nuanced verdict. | "
    "RETAIN (frontier-first delta demonstrated a fifth time: right cell "
    "selected, executed through the full gate stack, and closed with a "
    "decisive, independently-verified nuanced verdict) |\n"
)
rules_marker = "\n## Learning rules\n"
assert rules_marker in s, "learning rules marker not found"
s = s.replace(rules_marker, new_hist_row + rules_marker)

# 4) Update the "Next:" paragraph in the Active strategy delta section.
old_next_para = (
    "- **Next:** momentum is admitted as candidate positive evidence; the next "
    "frontier action is to test generalization beyond the short-horizon "
    "window that was swept — run longer-horizon momentum (lookback 20/60) "
    "through the same regime + independent-verification gates to determine "
    "whether the edge survives at the classic momentum horizon, before any "
    "broader generalization."
)
new_next_para = (
    "- **Next:** momentum is now closed — admitted as candidate positive "
    "evidence, qualified with the horizon finding (short-horizon momentum "
    "generalizes to lookback 20 and 60 in magnitude; the 60-day regime-"
    "stability verdicts are reported as null-dispersion-sensitive); the next "
    "frontier action is to verify the pre-existing cross-sectional-momentum "
    "artifact (lookback 63, FALSIFIED: 9/10 assets CONSISTENT_WITH_NOISE, "
    "perturbation CONSISTENT_WITH_NOISE, concentration NO_EDGE) with "
    "`verify_cross_sectional_momentum.py` and fold the verified verdict into "
    "the frontier, so the frontier records both sides of the momentum "
    "boundary on this universe (short-horizon per-asset momentum survives; "
    "63-day cross-sectional momentum does not)."
)
assert old_next_para in s, "Next paragraph not found"
s = s.replace(old_next_para, new_next_para)

open(p, "w").write(s)
print("LEARNING_STATE.md updated successfully; new length:", len(s))
