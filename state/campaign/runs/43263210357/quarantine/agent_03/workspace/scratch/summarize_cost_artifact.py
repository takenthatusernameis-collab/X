"""Compact summary of the cost-sensitivity artifact for durable recording."""
import json
from pathlib import Path

art = json.load(open(Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"))
print("signal_diagnostic_summary:", json.dumps(art["signal_diagnostic_summary"], sort_keys=True))
print("cross_consistency check:", art["cross_consistency_vs_prior"]["check"])
print("determinism:", art["determinism"])
for v in art["classification"]:
    for lb in art["classification"][v]:
        a = art["classification"][v][lb]
        print(f"variant={v} lb={lb} counts={a['counts']} decision={a['decision']} "
              f"mean_med={a['mean_candidate_median_by_cost']} "
              f"robust_assets={a['cost_robust_assets']}")
print("existing lb=5 per-asset:")
for t, pa in art["classification"]["existing"]["5"]["per_asset"].items():
    print("  ", t, pa["classification"], pa["mean_candidate_median"])
print("corrected lb=5 per-asset:")
for t, pa in art["classification"]["corrected"]["5"]["per_asset"].items():
    print("  ", t, pa["classification"], pa["mean_candidate_median"])