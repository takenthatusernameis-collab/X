#!/usr/bin/env python3
# Minimal reproduction of the comprehension issue

HOLD_PERIODS = (1, 2, 3, 5)

ticker_order = ['AAPL']
profiles = {
    'AAPL': {
        '1': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '2': {'verdict': 'REGIME_STABLE', 'medians': [0.054, 0.092, 0.069]},
        '3': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '5': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
    }
}

# Let's break it down step by step to understand the issue

print("Testing different comprehension structures...")

# 1. Simple comprehension that should work
hold_profiles_simple = {
    t: {
        str(h): "value"
        for h in HOLD_PERIODS
    }
    for t in ticker_order
}
print("Simple comprehension:", hold_profiles_simple)

# 2. Comprehension with conditional (the original working pattern)
hold_profiles_conditional = {
    t: {
        str(h): (
            "STABLE_EDGE" if True
            else "NO_EDGE"
        )
        for h in HOLD_PERIODS
    }
    for t in ticker_order
}
print("Conditional comprehension (working):", hold_profiles_conditional)

# 3. The problematic structure - let's see if we can isolate the issue
print("\nAttempting to reproduce the error...")

# Let's manually construct the exact pattern from the original file
# The issue might be with the nested conditional structure

def test_hold_profile_structure():
    print("Testing hold profile structure...")
    
    # Test the all() function specifically
    print("Testing all() function...")
    
    # This should work
    result = all([True, True, True])
    print(f"all([True, True, True]) = {result}")
    
    # This should also work
    result2 = all(True)
    print(f"all(True) = {result2}")  # This should fail - bool not iterable
    
    # The error "'bool' object is not iterable" happens when you call all() on a bool
    # This suggests that somewhere in the code, all() is being called on a boolean value
    
    # Let's test the specific condition that might be causing this
    h = 1
    verdict = profiles['AAPL'][str(h)]["verdict"]
    medians = profiles['AAPL'][str(h)]["medians"]
    
    condition = (
        verdict == 'REGIME_STABLE'
        and all(m > 0 for m in medians)
    )
    
    print(f"Condition for h={h}: verdict={verdict}, medians={medians}")
    print(f"Condition result: {condition}")
    
    # Now test with the all() function directly
    try:
        result3 = all(condition)
        print(f"all(condition) = {result3}")
    except Exception as e:
        print(f"ERROR calling all(condition): {e}")
    
    # This is the problematic line - calling all() on a boolean!
    try:
        result4 = all(condition, True)
        print(f"all(condition, True) = {result4}")
    except Exception as e:
        print(f"ERROR calling all(condition, True): {e}")

test_hold_profile_structure()