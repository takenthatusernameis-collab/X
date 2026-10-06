import json
with open("state/check_artifacts/breakout_20day_cont_results.json") as f:
    d = json.load(f)
print("dataset_id:", d["dataset_id"])
print("lookbacks:", d["lookbacks"], "canonical:", d["canonical_lookback"])
print("sensitivity_counts:", d["universe"]["sensitivity_counts"])
print("lookback_counts:", {str(k): v for k, v in d["universe"]["lookback_counts"].items()})
print("ticker_order:", d["universe"]["ticker_order"])
print("ARTIFACT VALID")
