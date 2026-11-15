import json
import numpy as np

art = json.load(open("state/check_artifacts/momentum_cost_sensitivity_results.json"))
for cost in ["zero", "realistic", "conservative"]:
    for lb in ["3", "5", "10"]:
        e = art["results"][cost][lb]["MSFT"]
        meds = e["medians"]
        nulls = e["null_medians"]
        cd = float(np.std(meds))
        nd = float(np.std(nulls))
        print(cost, "lb", lb, "medians", meds, "nulls", nulls,
              "stored_cd", e["candidate_dispersion"], "recomp_cd", round(cd, 4),
              "stored_nd", e["null_dispersion"], "recomp_nd", round(nd, 4),
              "verdict", e["verdict"])