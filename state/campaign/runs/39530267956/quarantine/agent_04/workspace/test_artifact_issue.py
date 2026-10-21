#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import json

# Mock data to reproduce the issue
profiles = {
    'AAPL': {
        '1': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '2': {'verdict': 'REGIME_STABLE', 'medians': [0.054, 0.092, 0.069]},
        '3': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '5': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
    }
}

try:
    # This is the problematic line from momentum_hold_period_sensitivity.py line 444-461
    hold_profiles = {t: {str(h): (
        "STABLE_EDGE" if all(
            profiles[t][str(h)]["verdict"] == "REGIME_STABLE"
            and all(m > 0 for m in profiles[t][str(h)]["medians"])
        ) else (
            "SINGULAR" if (
                sum(
                    profiles[t][str(h2)]["verdict"] == "REGIME_STABLE"
                    and all(m > 0 for m in profiles[t][str(h2)]["medians"])
                    for h2 in (1, 2, 3, 5)
                ) == 1
            ) else (
                "WEAKENS" if sum(
                    profiles[t][str(h2)]["verdict"] == "REGIME_STABLE"
                    and all(m > 0 for m in profiles[t][str(h2)]["medians"])
                    for h2 in (1, 2, 3, 5)
                ) > 1 else "NO_EDGE"
            )
        )
    ) for h in (1, 2, 3, 5)} for t in profiles}
    print("SUCCESS: hold_profiles created successfully")
    print(hold_profiles)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()