def test_syntax():
    # Test the problematic code pattern
    HOLD_PERIODS = (1, 2, 3, 5)
    
    # Test 1: Simple conditional expression
    condition1 = (False and True)
    print(f"Test 1: {condition1}")
    
    # Test 2: Nested conditional (what the script tries to do)
    profiles = {
        'AAPL': {
            '1': {'verdict': 'REGIME_STABLE', 'medians': [0.1, 0.2, 0.3]},
            '2': {'verdict': 'REGIME_STABLE', 'medians': [0.1, 0.2, 0.3]},
        }
    }
    ticker_order = ['AAPL']
    
    result = (
        "STABLE_EDGE" if (
            profiles['AAPL']['1']["verdict"] == "REGIME_STABLE" and
            all(m > 0 for m in profiles['AAPL']['1']["medians"])
        ) else (
            "SINGULAR" if (
                sum(
                    1 for h2 in HOLD_PERIODS
                    if profiles['AAPL'][str(h2)]["verdict"] == "REGIME_STABLE"
                    and all(m > 0 for m in profiles['AAPL'][str(h2)]["medians"])
                ) == 1
            ) else (
                "WEAKENS" if sum(
                    1 for h2 in HOLD_PERIODS
                    if profiles['AAPL'][str(h2)]["verdict"] == "REGIME_STABLE"
                    and all(m > 0 for m in profiles['AAPL'][str(h2)]["medians"])
                ) > 1 else "NO_EDGE"
            )
        )
    )
    
    print(f"Test 2 result: {result}")
    
    # Test 3: List comprehension with all() 
    stable = [t for t in ticker_order if all(
        (profiles[t][str(h)]["verdict"] == "REGIME_STABLE" and
         all(m > 0 for m in profiles[t][str(h)]["medians"]))
        for h in HOLD_PERIODS
    )]
    print(f"Test 3 stable: {stable}")

if __name__ == "__main__":
    test_syntax()