import json

files = [
    "state/activation_status.json",
    "state/check_artifacts/momentum_sweep_results.json",
    "state/check_artifacts/momentum_concentration.json",
]
for f in files:
    d = json.load(open(f))
    print(f, "OK - top keys:", list(d.keys())[:4])

# Spot-check key fields of the activation receipt
r = json.load(open("state/activation_status.json"))
assert r["activation_id"] == "37324143509"
assert r["status"] == "COMPLETE"
assert r["phase"] == "COMPLETE"
print("receipt fields OK: activation_id, status, phase")

c = json.load(open("state/check_artifacts/momentum_concentration.json"))
assert c["base"]["verdict"] == "DISTRIBUTED"
assert c["by_lookback"]["lookback_20"]["verdict"] == "CONCENTRATED"
assert c["determinism_r1_equals_r2"]
print("concentration artifact OK: base DISTRIBUTED, lookback_20 CONCENTRATED, deterministic")
