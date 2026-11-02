#!/usr/bin/env python3
import json
from pathlib import Path

artifact_path = Path('/home/runner/work/X/X/state/check_artifacts/momentum_cost_robustness_results.json')

with open(artifact_path) as f:
    artifact = json.load(f)

print('=== Artifact Structure ===')
print('Keys:', list(artifact.keys()))
print()

if 'universe' in artifact:
    print('Universe robustness_counts:', artifact['universe'].get('robustness_counts', 'NOT FOUND'))
    print('Universe ticker_order:', artifact['universe'].get('ticker_order', [])[:5])
    print()

if 'per_asset' in artifact:
    print('First few per_asset entries:')
    for ticker in ['AAPL', 'AMZN', 'NVDA']:
        if ticker in artifact['per_asset']:
            print(f'{ticker}:')
            for lb in ['3', '5', '10']:
                if lb in artifact['per_asset'][ticker]:
                    print(f'  lookback {lb}: verdict={artifact["per_asset"][ticker][lb]["verdict"]}, actual_median={artifact["per_asset"][ticker][lb]["actual_medians"][0]}')
    print()

print('Costs:', artifact.get('costs'))
