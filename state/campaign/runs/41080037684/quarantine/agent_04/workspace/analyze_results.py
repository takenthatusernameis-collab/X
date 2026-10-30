import json
with open('state/check_artifacts/momentum_cost_robustness_results.json') as f:
    data = json.load(f)

print('UNIVERSE-LEVEL VERDICT SUMMARY:')
verdict_counts = data['universe']['verdict_counts']
for verdict, count in verdict_counts.items():
    print(f'  {verdict}: {count}')

total = sum(verdict_counts.values())
positive = sum(count for v, count in verdict_counts.items() if v in ['REGIME_STABLE', 'REGIME_DEPENDENT'])
print(f'\nPositive evidence (REGIME_STABLE or REGIME_DEPENDENT): {positive}')
print(f'Total assets tested: {total}')
print(f'Positive evidence rate: {positive/total*100:.1f}%')
