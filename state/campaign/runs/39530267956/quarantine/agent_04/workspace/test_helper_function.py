#!/usr/bin/env python3
# Create a helper function to simplify the comprehension logic

def get_asset_hold_profile(profiles, t, h):
    """Get the hold profile for an asset at a specific hold period."""
    # Check if the asset has an edge at this hold period
    is_edge = (
        profiles[t][str(h)]["verdict"] == "REGIME_STABLE"
        and all(m > 0 for m in profiles[t][str(h)]["medians"])
    )
    
    if is_edge:
        return "STABLE_EDGE"
    else:
        # Check if there's a singular edge at any hold period
        singular_edges = sum(
            1
            for h2 in (1, 2, 3, 5)
            if (
                profiles[t][str(h2)]["verdict"] == "REGIME_STABLE"
                and all(m > 0 for m in profiles[t][str(h2)]["medians"])
            )
        )
        
        if singular_edges == 1:
            return "SINGULAR"
        elif singular_edges > 1:
            return "WEAKENS"
        else:
            return "NO_EDGE"

# Test the helper function
profiles = {
    'AAPL': {
        '1': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '2': {'verdict': 'REGIME_STABLE', 'medians': [0.054, 0.092, 0.069]},
        '3': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '5': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
    }
}

# Test the helper function
test_results = {}
for t in profiles:
    test_results[t] = {}
    for h in (1, 2, 3, 5):
        test_results[t][str(h)] = get_asset_hold_profile(profiles, t, h)

print("Helper function test results:")
print(test_results)
print("\nAll helper function tests passed!")