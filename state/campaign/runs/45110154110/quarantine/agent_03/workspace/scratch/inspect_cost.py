"""Read-only inspection of the momentum cost-sensitivity artifact.

Prints, per asset and lookback, the per-segment candidate medians and matched
null medians at zero / realistic / conservative cost, plus the candidate
median deltas (realistic - zero). This is the independent evidence check that
costs actually reduce the candidate medians (a COST_ROBUST label is only
credible if the medians degrade, not stay identical).
"""
import json
from pathlib import Path

ART = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
a = json.load(open(ART))

print("primary_verdict:", a["primary_verdict"])
print("zero_cost_cross_check:", a["zero_cost_cross_check"], "| determinism:", a["determinism"])
print("cost_grid:", a["cost_grid"])
print()

levels = ("zero", "realistic", "conservative")
for lb in ("3", "5", "10"):
    print("=== lookback {} ===".format(lb))
    print("  {:6s} {:>26s} {:>26s} {:>26s}  {:>8s}".format(
        "asset", "zero_med", "real_med", "cons_med", "d(real)"))
    for t in a["results"]["zero"][lb]:
        z = a["results"]["zero"][lb][t]["medians"]
        r = a["results"]["realistic"][lb][t]["medians"]
        c = a["results"]["conservative"][lb][t]["medians"]
        # mean candidate median per level; delta realistic - zero
        dz = sum(z) / len(z)
        dr = sum(r) / len(r)
        dc = sum(c) / len(c)
        print("  {:6s} {:>26s} {:>26s} {:>26s}  {:>+8.3f}".format(
            t, str(z), str(r), str(c), dr - dz))
        # also null means to confirm null moves with costs
    print()

# Aggregate candidate-median drag at lookback 5 (canonical)
print("=== lookback 5 aggregate (mean of per-asset mean candidate medians) ===")
for lv in levels:
    means = []
    for t in a["results"][lv]["5"]:
        meds = a["results"][lv]["5"][t]["medians"]
        means.append(sum(meds) / len(meds))
    print("  {:12s}: mean-of-means = {:+.4f}".format(lv, sum(means) / len(means)))