import json
d = json.load(open('state/check_artifacts/momentum_cost_sensitivity_results.json'))
for t in ['AAPL', 'NVDA', 'JPM', 'MSFT']:
    a = d['per_asset'][t]
    print('====', t, 'cost_robustness', a['cost_robustness'], 'survives', a['survives'])
    for lb in ['3', '5', '10']:
        for c in ['zero', 'realistic', 'conservative']:
            cell = a['cells'][lb][c]
            print('  lb', lb, c, 'medians', cell['medians'], 'nulls', cell['null_medians'], 'verdict', cell['verdict'], 'nfolds', cell['n_folds'])