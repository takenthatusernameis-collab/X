import json
d = json.load(open("/home/runner/work/X/X/state/check_artifacts/mean_reversion_results.json"))
print(json.dumps(d, indent=2, default=str))
