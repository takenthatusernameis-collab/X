import json
from pathlib import Path

p = Path("state/check_artifacts/mean_reversion_results.json")
d = json.load(open(p))
print("universe verdict_counts:", d["universe"]["verdict_counts"])
print("universe ticker_order:", d["universe"]["ticker_order"])
print("per_asset sample (AAPL):", d["per_asset"]["AAPL"])
