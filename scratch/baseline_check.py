import json, numpy as np
p = "/home/runner/work/X/X/state/check_artifacts/cross_sectional_momentum_results.json"
a = json.load(open(p))
ps = [("bottom_k",3),("lookback",63),("top_k",3)]
# find index of the baseline param set in artifact param_sets
idx = None
for j, ps_a in enumerate(a["perturbation"]["param_sets"]):
    if tuple(sorted([(k,v) for k,v in ps_a])) == tuple(sorted(ps)):
        idx = j; break
print("baseline param set index:", idx)
for name in ["calm","mean_reverting","trending","turbulent"]:
    print(name, "null_medians[", idx, "] =", a["perturbation"]["null_medians"][name][idx])
nulls = [a["perturbation"]["null_medians"][n][idx] for n in ["calm","mean_reverting","trending","turbulent"]]
print("median of stored null_medians at baseline:", round(float(np.median(nulls)),3))
print("artifact perturbation.baseline_null:", a["perturbation"]["baseline_null"])
cands = [a["perturbation"]["scenario_medians"][n][idx] for n in ["calm","mean_reverting","trending","turbulent"]]
print("artifact perturbation.baseline_candidate:", a["perturbation"]["baseline_candidate"],
      "| median of stored scenario_medians:", round(float(np.median(cands)),3))
