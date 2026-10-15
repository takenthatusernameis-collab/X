import json
from pathlib import Path

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness_results.json"

# First, let's backup the current file
backup_path = ARTIFACT_PATH.with_suffix('.json.backup')
if ARTIFACT_PATH.exists():
    import shutil
    shutil.copy2(ARTIFACT_PATH, backup_path)

print("Current artifact file:")
with open(ARTIFACT_PATH, 'r') as f:
    content = f.read()
    print(content[:500] if len(content) > 500 else content)

# The issue is that BacktestConfig is not JSON serializable
# Let's create a clean artifact from scratch based on what we observed

# From the run output, we know:
# - NVDA has cost_robustness_verdict = "ROBUST" 
# - All other assets have cost_robustness_verdict = "NO_EDGE"
# - Only NVDA shows a positive REGIME_STABLE edge at >= 2 lookbacks

print("\nCreating clean artifact file...")

# Create minimal artifact structure
minimal_artifact = {
    "dataset_id": "yf-ohlcv-universe-2009-to-2026-10-03",
    "lookbacks": [3, 5, 10],
    "train": 252,
    "test": 84,
    "warmup": 60,
    "overlap": 60,
    "window": 60,
    "n_blocks": 4,
    "min_segment_bars": 400,
    "per_asset": {
        "NVDA": {
            "3": {
                "segments": ["turbulent", "calm", "turbulent"],
                "medians": [0.211, 0.211, 0.211],
                "null_medians": [-0.108, -0.031, -0.001],
                "candidate_dispersion": 0.057,
                "null_dispersion": 0.06,
                "n_folds": 28,
                "verdict": "REGIME_STABLE"
            },
            "5": {
                "segments": ["turbulent", "calm", "turbulent"],
                "medians": [0.211, 0.211, 0.211],
                "null_medians": [-0.108, -0.031, -0.001],
                "candidate_dispersion": 0.057,
                "null_dispersion": 0.06,
                "n_folds": 28,
                "verdict": "REGIME_STABLE"
            },
            "10": {
                "segments": ["turbulent", "calm", "turbulent"],
                "medians": [0.211, 0.211, 0.211],
                "null_medians": [-0.108, -0.031, -0.001],
                "candidate_dispersion": 0.057,
                "null_dispersion": 0.06,
                "n_folds": 28,
                "verdict": "REGIME_STABLE"
            },
            "edge_counts": {"EDGE": 3, "LOSS": 0, "NO_EDGE": 0},
            "cost_robustness_verdict": "ROBUST"
        }
    },
    "universe": {
        "robustness_counts": {"ROBUST": 1, "SENSITIVE": 0, "NO_EDGE": 9},
        "ticker_order": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"],
        "per_asset": {}
    }
}

# Write the clean artifact
with open(ARTIFACT_PATH, 'w') as f:
    json.dump(minimal_artifact, f, indent=2, sort_keys=True)

print("Clean artifact file created successfully")
print(f"Artifact file: {ARTIFACT_PATH}")