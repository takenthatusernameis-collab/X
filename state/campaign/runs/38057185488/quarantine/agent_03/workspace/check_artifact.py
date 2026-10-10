#!/usr/bin/env python3
import json

with open('/home/runner/work/X/X/state/check_artifacts/momentum_cost_sensitivity_results.json', 'r') as f:
    artifact = json.load(f)

# Look at AMZN data for lookback 5 (the one being tested)
amzn_data = artifact['lookback_results']['5']
print('AMZN data structure:')
for cfg in ['zero_cost', 'cost']:
    print(f'\n{cfg}:')
    for ticker in ['AMZN', 'JPM']:
        print(f'  {ticker}: medians={amzn_data[cfg][ticker]["medians"]}, verdict={amzn_data[cfg][ticker]["verdict"]}')