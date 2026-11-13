"""Summarize the cost-sensitivity artifact (read-only)."""
import json
from pathlib import Path

p = Path("state/check_artifacts/momentum_cost_sensitivity_results.json")
a = json.loads(p.read_text())

for v in a["variants"]:
    print("=== variant:", v)
    for lb in ("3", "5", "10"):
        print(" lookback", lb, "counts:", a["universe"][v][lb]["counts"])
        if v == "corrected_momentum" and lb == "10":
            for t in a["universe"][v][lb]["ticker_order"]:
                row = a["per_variant"][v][lb][t]
                meds = {cl: row[cl]["medians"] for cl in a["cost_levels"]}
                print("   ", t, row["_cost_robustness"],
                      "zero_medians=", meds["zero"],
                      "stress_medians=", meds["stress"],
                      "verdicts=", {cl: row[cl]["verdict"] for cl in a["cost_levels"]})

print("\nExisting (constant long) framework verdicts at lookback 5:")
for t in a["universe"]["existing_momentum"]["5"]["ticker_order"]:
    row = a["per_variant"]["existing_momentum"]["5"][t]
    print("  ", t, row["_cost_robustness"],
          {cl: row[cl]["verdict"] for cl in a["cost_levels"]})
print("\nTurnover:", a["turnover_profile"])