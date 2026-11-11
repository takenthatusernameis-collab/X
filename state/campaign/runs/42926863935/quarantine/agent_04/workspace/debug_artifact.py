from pathlib import Path
import json

# Load the existing lookback sweep artifact
path = Path("/home/runner/work/X/X/state/check_artifacts/momentum_lookback_sweep_results.json")
with open(path, "r") as f:
    artifact = json.load(f)

print("Keys in artifact:", list(artifact.keys()))
print("\nStructure of AMZN:", list(artifact["per_asset"]["AMZN"].keys()))

# Check what TARGET assets contain
target = ["AMZN", "JPM"]
for ticker in target:
    print(f"\n{ticker} structure:", list(artifact["per_asset"][ticker].keys()))
    print(f"  edge_counts: {artifact['per_asset'][ticker]['edge_counts']}")
    print(f"  robustness_verdict: {artifact['per_asset'][ticker]['robustness_verdict']}")
    print(f"  lookback_medians keys: {list(artifact['per_asset'][ticker]['lookback_medians'].keys())}")
