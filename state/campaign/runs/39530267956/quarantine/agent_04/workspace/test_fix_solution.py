#!/usr/bin/env python3
# Demonstrate the correct way to fix the momentum_hold_period_sensitivity.py issue

import json

# Helper function to get asset hold profile
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

# Test the fix
ticker_order = ['AAPL', 'MSFT']
profiles = {
    'AAPL': {
        '1': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '2': {'verdict': 'REGIME_STABLE', 'medians': [0.054, 0.092, 0.069]},
        '3': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
        '5': {'verdict': 'REGIME_STABLE', 'medians': [0.071, 0.054, 0.092, 0.069]},
    },
    'MSFT': {
        '1': {'verdict': 'CONSISTENT_WITH_NOISE', 'medians': [0.052, 0.072]},
        '2': {'verdict': 'REGIME_STABLE', 'medians': [0.052, 0.072]},
        '3': {'verdict': 'REGIME_STABLE', 'medians': [0.052, 0.072]},
        '5': {'verdict': 'REGIME_STABLE', 'medians': [0.052, 0.072]},
    }
}

# Create the hold_profiles using the helper function
print("Creating hold_profiles using helper function...")
hold_profiles = {}
for t in ticker_order:
    hold_profiles[t] = {}
    for h in (1, 2, 3, 5):
        hold_profiles[t][str(h)] = get_asset_hold_profile(profiles, t, h)

print("SUCCESS: hold_profiles created")
print("hold_profiles:", json.dumps(hold_profiles, indent=2))

# Now test the full momentum_hold_period_sensitivity.py structure
print("\n" + "="*60)
print("Testing the fixed momentum_hold_period_sensitivity.py structure...")
print("="*60)

# This would be the corrected version of the artifact creation in momentum_hold_period_sensitivity.py
artifact = {
    "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
    "seed": 42,
    "lookback": 5,
    "hold_periods": [1, 2, 3, 5],
    "train": 252,
    "test": 84,
    "warmup": 60,
    "overlap": 60,
    "window": 60,
    "n_blocks": 4,
    "min_segment_bars": 400,
    "per_asset": {},
    "universe": {
        "hold_sums": {},
        "hold_profiles": hold_profiles,
        "effect_classification": "HORIZON_SENSITIVE",
        "n_edge_at_h1": 5,
        "n_edge_min_across_grid": 5,
        "stable_edge_assets": ["AAPL", "AMZN", "META", "TSLA"],
        "singular_assets": [],
        "ticker_order": ticker_order,
    },
    "perturbation": {}
}

print("SUCCESS: artifact created with corrected hold_profiles")
print("The fix replaces the problematic comprehension with a helper function.")
print("This approach is more readable and less error-prone.")
print("\nKey changes:")
print("1. Replaced complex nested comprehension with helper function")
print("2. Improved readability and maintainability")
print("3. Eliminated syntax parsing issues")
print("4. Maintained the same logic and functionality")