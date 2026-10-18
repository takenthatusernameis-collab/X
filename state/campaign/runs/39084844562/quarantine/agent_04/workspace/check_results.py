import json

with open('/home/runner/work/X/X/state/check_artifacts/momentum_hold_period_sensitivity_results.json', 'r') as f:
    data = json.load(f)

print('Dataset ID:', data['dataset_id'])
print('Lookback:', data['lookback'])
print('Hold periods:', data['hold_periods'])
print('Edge classification:', data['universe']['effect_classification'])
print('Number of edge assets at H=1:', data['universe']['n_edge_at_h1'])
print('Minimum edges across grid:', data['universe']['n_edge_min_across_grid'])
print('Stable edge assets:', data['universe']['stable_edge_assets'])