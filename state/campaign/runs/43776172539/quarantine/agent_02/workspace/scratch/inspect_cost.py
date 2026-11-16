import json
from pathlib import Path

p = Path("state/check_artifacts/momentum_cost_sensitivity_results.json")
a = json.loads(p.read_text())
print("overall:", a["overall"])
for lb in ["3", "5", "10"]:
    cell = a["cells"][lb]
    print(f"\n=== lookback {lb} : {cell['classification']} cost_drag={cell['cost_drag']}")
    for lvl in ["zero", "realistic", "conservative"]:
        m = cell["per_cost_level"][lvl]["metrics"]
        print(f"  {lvl:12s} cand={m['mean_candidate_median']:+.4f} "
              f"null={m['mean_null_median']:+.4f} gap={m['mean_candidate_minus_null']:+.4f} "
              f"edges={m['edge_asset_count']} beat_null={m['beat_null_asset_count']}")
    print("  per-asset (cand zero -> conservative, delta | null zero -> conservative, delta | cons verdict):")
    z = cell["per_cost_level"]["zero"]["per_asset"]
    c = cell["per_cost_level"]["conservative"]["per_asset"]
    for t in z:
        cz = sum(z[t]["medians"]) / len(z[t]["medians"])
        cc = sum(c[t]["medians"]) / len(c[t]["medians"])
        nz = sum(z[t]["null_medians"]) / len(z[t]["null_medians"])
        nc = sum(c[t]["null_medians"]) / len(c[t]["null_medians"])
        print(f"    {t:6s} cand {cz:+.4f}->{cc:+.4f} ({cc-cz:+.4f}) | "
              f"null {nz:+.4f}->{nc:+.4f} ({nc-nz:+.4f}) | {c[t]['verdict']}")