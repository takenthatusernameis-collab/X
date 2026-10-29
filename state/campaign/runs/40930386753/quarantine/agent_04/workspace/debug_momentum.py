import json

data = json.load(open('state/check_artifacts/momentum_results.json'))
print('Top-level keys:', list(data.keys()))
print()
print('Universe keys:', list(data['universe'].keys()))
print('Universe per_asset keys:', list(data['universe']['per_asset'].keys())[:3])
print('First ticker (AAPL) keys:', list(data['universe']['per_asset']['AAPL'].keys()))
print('AAPL medians:', data['universe']['per_asset']['AAPL']['medians'][:2])