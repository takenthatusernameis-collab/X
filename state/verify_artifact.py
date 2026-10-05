import json
d = json.load(open("state/check_artifacts/momentum_results.json"))
assert d["dataset_id"] == "yf-ohlcv-universe-2009-to-2026-10-03"
assert d["lookback"] == 5
vc = d["universe"]["verdict_counts"]
print("artifact reload OK")
print("dataset:", d["dataset_id"], "| lookback:", d["lookback"],
      "| verdict_counts:", vc)
n_stable = vc["REGIME_STABLE"] + vc["REGIME_STABLE_LOSS"]
print("assets with a regime verdict (STABLE/STABLE_LOSS):", n_stable)
print("positive-REGIME_STABLE medians:",
      [v["medians"] for k, v in d["universe"]["per_asset"].items()
       if v["verdict"] == "REGIME_STABLE"])
