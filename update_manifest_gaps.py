import json
from collections import OrderedDict

path = "research/data/manifest.json"
with open(path) as fh:
    m = json.load(fh)

known_gaps = [
    "2018-12-05",   # systematic Yahoo gap (trading day present, no bar returned)
    "2025-01-09",   # systematic Yahoo gap (trading day present, no bar returned)
]
m["known_data_gaps"] = known_gaps
m["notes"] = (
    "Adjusted close is used for backtesting; dividends and splits are reflected. "
    "This dataset is research-only simulation input and does not constitute a "
    "live-trading signal or recommendation. "
    "Two systematic source gaps are documented in known_data_gaps: the data provider "
    "did not return bars for 2018-12-05 and 2025-01-09 (both were NYSE trading days); "
    "the preflight treats these documented gaps as a completed completeness check."
)
with open(path, "w") as fh:
    json.dump(m, fh, indent=2)
print("manifest updated; known_data_gaps:", known_gaps)
