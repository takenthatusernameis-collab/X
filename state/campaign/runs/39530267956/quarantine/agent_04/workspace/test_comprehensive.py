#!/usr/bin/env python3
# Test complex comprehension structure

print("Testing comprehension structure...")

# The problematic structure from momentum_hold_period_sensitivity.py
ticker_order = ['AAPL']
profiles = {
    'AAPL': {
        '1': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '2': {'verdict': 'REGIME_STABLE', 'medians': [0.054, 0.092, 0.069]},
        '3': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '5': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
    }
}

# Test 1: Simple comprehension (should work)
print("Test 1: Simple comprehension")
result1 = {str(h): 'value' for h in (1, 2, 3, 5)}
print("  Result:", result1)

# Test 2: Comprehension with simple conditional (should work)
print("Test 2: Comprehension with simple conditional")
result2 = {
    str(h): (
        'A' if True
        else 'B'
    )
    for h in (1, 2, 3, 5)
}
print("  Result:", result2)

# Test 3: Comprehension with all() function (should work)
print("Test 3: Comprehension with all() function")
result3 = {
    str(h): (
        'A' if all([True, True])
        else 'B'
    )
    for h in (1, 2, 3, 5)
}
print("  Result:", result3)

# Test 4: More complex structure similar to momentum_hold_period_sensitivity.py
print("Test 4: Complex nested structure")
try:
    result4 = {t: {
        str(h): (
            "STABLE_EDGE" if all(
                profiles[t][str(h)]["verdict"] == "REGIME_STABLE"
                and all(m > 0 for m in profiles[t][str(h)]["medians"])
            ) else (
                "SINGULAR" if (
                    sum(
                        (profiles[t][str(h2)]["verdict"] == "REGIME_STABLE"
                         and all(m > 0 for m in profiles[t][str(h2)]["medians"]))
                        for h2 in (1, 2, 3, 5)
                    ) == 1
                ) else (
                    "WEAKENS" if sum(
                        (profiles[t][str(h2)]["verdict"] == "REGIME_STABLE"
                         and all(m > 0 for m in profiles[t][str(h2)]["medians"]))
                        for h2 in (1, 2, 3, 5)
                    ) > 1 else "NO_EDGE"
                )
            )
        )
        for h in (1, 2, 3, 5)
    } for t in ticker_order}
    print("  Result:", result4)
except Exception as e:
    print("  ERROR:", e)

print("All tests completed!")