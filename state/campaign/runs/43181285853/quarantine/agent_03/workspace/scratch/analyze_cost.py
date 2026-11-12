import json
import numpy as np

with open("state/check_artifacts/momentum_cost_sensitivity_results.json") as f:
    art = json.load(f)

for lb in ["3", "5", "10"]:
    print("=== lookback", lb, "===")
    print("{:6s} {:>12s} {:>12s} {:>12s} {:>12s} {:>10s}".format(
        "tkr", "zero_ret", "real_ret", "cons_ret", "real_drag", "cons_drag"))
    zr, rr, cr = [], [], []
    for t, sr in art["per_asset"]["zero"][lb].items():
        z = sr["full_sample"]["total_return"]
        r = art["per_asset"]["realistic"][lb][t]["full_sample"]["total_return"]
        c = art["per_asset"]["conservative"][lb][t]["full_sample"]["total_return"]
        zr.append(z); rr.append(r); cr.append(c)
        print("{:6s} {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:10.4f}".format(
            t, z, r, c, z - r, z - c))
    print("mean   {:12.4f} {:12.4f} {:12.4f} {:12.4f} {:10.4f}".format(
        np.mean(zr), np.mean(rr), np.mean(cr), np.mean(np.array(zr) - np.array(rr)),
        np.mean(np.array(zr) - np.array(cr))))
    # median segment-median retained
    for cost in ["realistic", "conservative"]:
        vals = []
        for t in art["per_asset"]["zero"][lb]:
            b = np.mean(art["per_asset"]["zero"][lb][t]["medians"])
            m = np.mean(art["per_asset"][cost][lb][t]["medians"])
            vals.append(m / b)
        print("  retained median-segment fraction {}: mean {:.4f} min {:.4f} max {:.4f}".format(
            cost, np.mean(vals), np.min(vals), np.max(vals)))
    print()