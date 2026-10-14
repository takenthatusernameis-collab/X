#!/usr/bin/env python3
import json
from pathlib import Path

def check_artifact_structure():
    # Check if artifacts exist
    cost_artifact_path = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
    mom_artifact_path = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"
    
    print(f"Checking artifacts in {Path.cwd()}")
    print(f"Cost artifact exists: {cost_artifact_path.exists()}")
    print(f"Momentum artifact exists: {mom_artifact_path.exists()}")
    
    if cost_artifact_path.exists():
        with open(cost_artifact_path) as f:
            cost_data = json.load(f)
        
        print(f"\nCost artifact keys: {list(cost_data.keys())}")
        
        if 'lookback_results' in cost_data:
            lookback_results = cost_data['lookback_results']
            print(f"Lookback keys in cost artifact: {list(lookback_results.keys())}")
            
            for lb in ['3', '5', '10']:
                if lb in lookback_results:
                    print(f"\nLookback {lb} structure:")
                    for ticker in ['AAPL', 'AMZN', 'TSLA'][:3]:  # Check first few tickers
                        if ticker in lookback_results[lb]:
                            entry = lookback_results[lb][ticker]
                            print(f"  {ticker}: type={type(entry)}, length={len(entry) if hasattr(entry, '__len__') else 'N/A'}")
                            if isinstance(entry, list) and len(entry) > 0:
                                print(f"    First entry keys: {list(entry[0].keys())}")
                            else:
                                print(f"    Entry content: {entry}")
    
    if mom_artifact_path.exists():
        with open(mom_artifact_path) as f:
            mom_data = json.load(f)
        
        print(f"\nMomentum artifact keys: {list(mom_data.keys())}")
        if 'universe' in mom_data:
            print(f"Momentum universe keys: {list(mom_data['universe'].keys())}")

if __name__ == "__main__":
    check_artifact_structure()