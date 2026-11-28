import json
d = json.load(open('state/check_artifacts/momentum_cost_sensitivity_results.json'))
u = d['universe']
print('class_counts', u['class_counts'])
print('lb5_survives_conservative', u['lb5_survives_conservative'])
print('candidate_positive_zero', u['candidate_positive_zero'])
print('candidate_positive_conservative', u['candidate_positive_conservative'])
print('candidate_eroded_by_cost', u['candidate_eroded_by_cost'])
print('determinism', d['determinism'])
print('cost_levels', d['cost_levels'], 'lookbacks', d['lookbacks'])
# per-asset verdict + shift
for t in u['ticker_order']:
    a = d['per_asset'][t]
    print(t, a['cost_robustness'], 'survives', a['survives'], 'pos', a['survives_pos'], 'shift', a['max_candidate_median_shift'], 'lb5cons', a['lb5_survives_conservative'])