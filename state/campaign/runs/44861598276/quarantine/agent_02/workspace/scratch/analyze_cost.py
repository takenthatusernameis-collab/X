import json
from pathlib import Path

p = Path("state/check_artifacts/momentum_cost_sensitivity_results.json")
a = json.load(open(p))
print("aggregate:", a["universe"]["aggregate_cost_robustness"])
print("edge_counts_l5:", a["universe"]["positive_edge_counts_l5"])
print()
for t in a["universe"]["ticker_order"]:
    print("ASSET", t)
    for lb in ["3", "5", "10"]:
        e = a["per_asset"][t]["lookbacks"][lb]
        print("  lookback", lb, "class", e["cost_robustness"])
        for lvl in ["zero", "realistic", "stress"]:
            sr = e[lvl]
            print("    {:9s} verdict={:22s} med={} null={} cdisp={} ndisp={}".format(
                lvl, sr["verdict"], sr["medians"], sr["null_medians"],
                sr["candidate_dispersion"], sr["null_dispersion"]))
    print()