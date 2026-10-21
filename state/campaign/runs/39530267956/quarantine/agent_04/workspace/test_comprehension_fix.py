#!/usr/bin/env python3
import sys

# Let's test the original problematic structure more carefully
# Simulating the exact problematic line structure

# This is what the comprehension should look like:
# {str(h): value for h in HOLD_PERIODS}
# NOT:
# {str(h): (
#     "STABLE_EDGE" if all(...) else ...
# ) for h in HOLD_PERIODS} - the for clause should be outside

# The correct pattern for a dictionary comprehension with complex conditional logic:
HOLD_PERIODS = (1, 2, 3, 5)

ticker_order = ['AAPL', 'MSFT']
profiles = {
    'AAPL': {
        '1': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '2': {'verdict': 'REGIME_STABLE', 'medians': [0.054, 0.092, 0.069]},
        '3': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '5': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
    },
    'MSFT': {
        '1': {'verdict': 'REGIME_STABLE', 'medians': [0.052, 0.072]},
        '2': {'verdict': 'REGIME_STABLE', 'medians': [0.052, 0.072]},
        '3': {'verdict': 'CONSISTENT_WITH_NOISE', 'medians': [0.052, 0.072]},
        '5': {'verdict': 'REGIME_STABLE', 'medians': [0.052, 0.072]},
    }
}

# The correct structure - the for loop is outside the conditional expression:
hold_profiles = {
    t: {
        str(h): (
            "STABLE_EDGE" if all(
                profiles[t][str(h)]["verdict"] == "REGIME_STABLE"
                and all(m > 0 for m in profiles[t][str(h)]["medians"])
            ) else (
                "SINGULAR" if (
                    sum(
                        (profiles[t][str(h2)]["verdict"] == "REGIME_STABLE"
                         and all(m > 0 for m in profiles[t][str(h2)]["medians"]))
                        for h2 in HOLD_PERIODS
                    ) == 1
                ) else (
                    "WEAKENS" if sum(
                        (profiles[t][str(h2)]["verdict"] == "REGIME_STABLE"
                         and all(m > 0 for m in profiles[t][str(h2)]["medians"]))
                        for h2 in HOLD_PERIODS
                    ) > 1 else "NO_EDGE"
                )
            )
        )
        for h in HOLD_PERIODS
    }
    for t in ticker_order
}

print("SUCCESS: hold_profiles created with correct structure")
print("AAPL:", hold_profiles['AAPL'])
print("MSFT:", hold_profiles['MSFT'])