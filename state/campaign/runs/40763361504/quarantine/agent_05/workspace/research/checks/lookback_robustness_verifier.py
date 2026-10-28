"""Lookback-robustness classifier for cost sensitivity verification.

This module provides the same robustness verdict logic as momentum_lookback_sweep.py,
but is re-implemented independently for verification purposes in the cost sensitivity
check. It matches the exact decision rules:

- For each lookback, classify as:
  - "EDGE" : REGIME_STABLE with uniformly positive medians (> 0)
  - "LOSS" : REGIME_STABLE_LOSS (uniformly negative medians, stable losses)
  - "NO_EDGE" : everything else (CONSISTENT_WITH_NOISE, REGIME_DEPENDENT, or REGIME_STABLE with non-uniform signs)

- Aggregate across lookbacks:
  - "ROBUST" : uniformly positive REGIME_STABLE edges at >= 2 lookbacks
  - "SENSITIVE" : a positive REGIME_STABLE edge at exactly one lookback only
  - "NO_EDGE" : no lookback shows a positive REGIME_STABLE edge
"""

LOOKBACKS = (3, 5, 10)


def lookback_verdict(sr):
    """Robustness verdict for one lookback's scenario result (independent reimplementation)."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset (independent reimplementation)."""
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts


# Simple test if run directly
if __name__ == "__main__":
    # Example test case
    sample_results = {
        "3": {"verdict": "REGIME_STABLE", "medians": [0.05, 0.06, 0.07]},
        "5": {"verdict": "REGIME_STABLE", "medians": [0.03, 0.04, 0.02]},
        "10": {"verdict": "REGIME_STABLE", "medians": [-0.01, -0.02, -0.01]},
    }
    verdict, counts = robustness_verdict(sample_results)
    print(f"Sample result: verdict={verdict}, counts={counts}")
    assert verdict == "ROBUST"
    assert counts["EDGE"] == 2
    assert counts["LOSS"] == 0
    assert counts["NO_EDGE"] == 1
    print("Test passed: Basic robustness classification works correctly")
