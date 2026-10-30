import json
with open('/home/runner/work/X/X/state/check_artifacts/momentum_cost_robustness_results.json') as f:
    data = json.load(f)

print('Input data:')
verdict_counts = data['universe']['verdict_counts']
print(f'Verdict counts: {verdict_counts}')
print(f'Total: {sum(verdict_counts.values())}')
print()

print('How script calculates positive evidence:')
positive_evidence = sum(1 for v in verdict_counts.keys() 
                       if v in ["REGIME_STABLE", "REGIME_DEPENDENT"])
print(f'Positive evidence (unique verdict types): {positive_evidence}')
print()

print('Check what the script actually counts:')
total_assets = len(data['per_asset'])
print(f'Total assets: {total_assets}')
positive_assets = 0
for asset, asset_data in data['per_asset'].items():
    lookback_results = asset_data['lookback_results']
    for lb, lb_data in lookback_results.items():
        verdict = lb_data['verdict']
        if verdict in ['REGIME_STABLE', 'REGIME_DEPENDENT']:
            positive_assets += 1
print(f'Positive (asset-lookback combos): {positive_assets} out of {total_assets * 3}')
print(f'(Note: each asset has 3 lookbacks: 3, 5, 10)')
print()

print('UNIVERSE-LEVEL VERDICT ANALYSIS:')
for verdict, count in verdict_counts.items():
    print(f'  {verdict}: {count}')

total = sum(verdict_counts.values())
positive_unique = sum(count for v, count in verdict_counts.items() if v in ['REGIME_STABLE', 'REGIME_DEPENDENT'])
print(f'\nUNIVERSE-LEVEL SUMMARY:')
print(f'Total verdict entries: {total}')
print(f'Positive verdict entries (REGIME_STABLE + REGIME_DEPENDENT): {positive_unique}')
print(f'Positive rate: {positive_unique/total*100:.1f}%')