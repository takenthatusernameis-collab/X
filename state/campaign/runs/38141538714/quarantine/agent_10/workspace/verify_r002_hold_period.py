import json

# Check if the main momentum artifact exists (baseline for H=1)
import os
artifact_path = "state/check_artifacts/momentum_results.json"
hold_path = "state/check_artifacts/momentum_hold_period_sensitivity_results.json"

print("=== Artifact Verification for R-002 (Holding Period Sensitivity) ===")

# 1. Verify main momentum baseline (H=1)
if os.path.exists(artifact_path):
    with open(artifact_path) as f:
        main_artifact = json.load(f)
    
    verdict_counts = main_artifact.get("universe", {}).get("verdict_counts", {})
    h1_edges = verdict_counts.get("REGIME_STABLE", 0) + verdict_counts.get("REGIME_STABLE_LOSS", 0)
    
    print(f"1. Baseline momentum (H=1) - Main artifact verified:")
    print(f"   - Assets with regime verdict: {h1_edges}")
    print(f"   - Positive edges: {verdict_counts.get('REGIME_STABLE', 0)}")
    
    if h1_edges >= 7:
        print(f"   ✓ Momentum has qualified evidence: {h1_edges} assets with regime verdict")
    else:
        print(f"   ✗ Insufficient qualified evidence: only {h1_edges} assets")
        
else:
    print(f"1. ✗ Baseline momentum artifact not found at {artifact_path}")
    
# 2. Check for hold-period results
if os.path.exists(hold_path):
    with open(hold_path) as f:
        hold_artifact = json.load(f)
    
    h1_edges = hold_artifact.get("universe", {}).get("n_edge_at_h1", 0)
    min_across_grid = hold_artifact.get("universe", {}).get("n_edge_min_across_grid", 0)
    effect_class = hold_artifact.get("universe", {}).get("effect_classification", "UNKNOWN")
    singular = len(hold_artifact.get("universe", {}).get("singular_assets", []))
    stable = hold_artifact.get("universe", {}).get("stable_edge_assets", [])
    
    print(f"\n2. Holding-period sensitivity results (R-002):")
    print(f"   - H=1 positive edges: {h1_edges}")
    print(f"   - Min across grid: {min_across_grid}")
    print(f"   - Effect classification: {effect_class}")
    print(f"   - Singular assets: {singular}")
    print(f"   - Stable edge assets: {stable}")
    
    # Success criteria
    if effect_class == "HORIZON_STABLE":
        print(f"   ✓ SUCCESS: HORIZON_STABLE - Edge persists across all holds")
        print(f"     Confirmed durable momentum (not a single-point timing artifact)")
    elif effect_class == "HORIZON_SENSITIVE":
        print(f"   ⚠ HORIZON_SENSITIVE - Edge present but varies with hold period")
        print(f"     Confirmed sensitivity to holding period")
    elif effect_class == "HORIZON_SINGULAR":
        print(f"   ✗ HORIZON_SINGULAR - Edge is a single-point timing artifact")
        print(f"     Momentum is not robust")
    elif effect_class == "HORIZON_DESTROYED":
        print(f"   ✗ HORIZON_DESTROYED - No edge survives any hold")
        print(f"     Momentum is not robust")
    else:
        print(f"   ? UNKNOWN classification")
        
else:
    print(f"\n2. ✗ Hold-period sensitivity artifact not found at {hold_path}")

print(f"\n=== Summary ===")
print("R-002 mission objective:")
print("  Test fixed lookback-5 momentum with predeclared hold-period grid")
print("  Determine if momentum effect survives without being a timing artifact")
print("  Scope: 10-asset universe, HOLD_PERIODS = (1, 2, 3, 5)")
print("")
print("Key findings:")
if os.path.exists(hold_path):
    data = json.load(open(hold_path))
    h1 = data.get("universe", {}).get("n_edge_at_h1", 0)
    effect_class = data.get("universe", {}).get("effect_classification", "UNKNOWN")
    if effect_class in ["HORIZON_STABLE", "HORIZON_SENSITIVE"]:
        print(f"  ✓ Momentum has positive edges across holding periods")
        print(f"  ✓ Effect classification: {effect_class}")
        print(f"  ✓ Success criterion achieved")
    else:
        print(f"  ✗ Effect classification: {effect_class}")
        print(f"  ✗ Success criterion not met")
else:
    print(f"  ✗ Results file not available - verification incomplete")