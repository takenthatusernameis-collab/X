import json
with open('state/check_artifacts/regime_filtered_momentum_results.json') as f:
    a = json.load(f)
print('universe keys:', list(a['universe'].keys()))
for var in a['universe']:
    print(var, a['universe'][var]['verdict_counts'])
print()
print('AMZN:', [a['universe'][var]['per_asset']['AMZN']['medians'] for var in a['universe']])
print('JPM:', [a['universe'][var]['per_asset']['JPM']['medians'] for var in a['universe']])
print('base per-asset:', [(t, d['medians'], d['verdict']) for t, d in a['universe']['base']['per_asset'].items()])
print('perturbation:', a['perturbation'])
