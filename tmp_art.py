import json
a = json.load(open("/home/runner/work/X/X/state/check_artifacts/cross_sectional_momentum_results.json"))
print({k: a[k] for k in a})
