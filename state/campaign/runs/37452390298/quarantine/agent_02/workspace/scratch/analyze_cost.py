import json
a = json.load(open('state/check_artifacts/momentum_cost_sensitivity_results.json'))
for lb in ['lookback_3','lookback_5','lookback_10']:
    print('====', lb)
    for lvl in ['zero','realistic','conservative','heavy']:
        rows = a['grid_results'][lb][['zero','realistic','conservative','heavy'].index(lvl)]['per_asset']
        print('  ', lvl, '->', [(r['ticker'], [round(m,3) for m in r['medians']],
                [round(n,3) for n in r['null_medians']], r['verdict']) for r in rows])
    print()
