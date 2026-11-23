"""Summarize the corrected momentum cost-sensitivity artifact for the record."""
import json
from pathlib import Path

ART = Path(__file__).resolve().parent.parent / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
art = json.loads(ART.read_text())

print("cost points:", art["cost_points"])
print()
print("=== corrected momentum: zero-cost medians & verdicts (lookbacks 3/5/10) ===")
print(f"{'ticker':6s} {'lb':>3s} {'medians':40s} {'verdict':22s} class")
for t in art["summary"]["ticker_order"]:
    for lb in ("3", "5", "10"):
        cell = art["per_asset"][t]["cost_cells"][lb]
        print(f"{t:6s} {lb:>3s} {str(cell['zero']['medians']):40s} "
              f"{cell['zero']['verdict']:22s} {cell['class']}")
    print()

print("=== reference lookback 5: candidate vs null vs cost point ===")
for t in art["summary"]["ticker_order"]:
    cell = art["per_asset"][t]["cost_cells"]["5"]
    print(f"{t:6s}")
    for cp in ("zero", "realistic", "stress2x"):
        c = cell[cp]
        gaps = [round(m - n, 3) for m, n in zip(c["medians"], c["null_medians"])]
        print(f"    {cp:10s} med={str(c['medians']):34s} null={str(c['null_medians']):34s} "
              f"gap={gaps} {c['verdict']}")
    print()

print("=== summary block ===")
print(json.dumps(art["summary"]["class_counts_all_lookbacks"], indent=2))
print("ref lookback zero-edge:", art["summary"]["reference_lookback_zero_edge_assets"])
print("ref lookback realistic-edge:", art["summary"]["reference_lookback_realistic_edge_assets"])
print("cross-consistency all_match:", art["cross_consistency"]["all_match"])