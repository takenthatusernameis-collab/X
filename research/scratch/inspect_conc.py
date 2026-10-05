import json, numpy as np
c = json.load(open("state/check_artifacts/momentum_concentration.json"))
for lb in c["lookbacks"]:
    out = c["by_lookback"]["lookback_{}".format(lb)]
    print("=== lookback_{} ===".format(lb))
    for t, e in out["effects"].items():
        print("  {:6s} signed={:+8.3f} abs={:+8.3f} mean_abs={:+8.4f}".format(
            t, e["signed_edge"], e["abs_effect"], e["mean_abs_effect"]))
