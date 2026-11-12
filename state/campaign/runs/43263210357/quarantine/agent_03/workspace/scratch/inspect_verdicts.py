import json
from pathlib import Path
art = json.load(open(Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"))
for v, lb, t in (("existing", "5", "MSFT"), ("existing", "3", "MSFT"),
                 ("corrected", "5", "NVDA"), ("corrected", "3", "NVDA")):
    print(f"--- {v} lb={lb} {t} ---")
    for lvl in ("zero", "realistic", "conservative"):
        c = art["cells"][v][lb][lvl][t]
        print(f"  {lvl:12s} verdict={c['verdict']:22s} disp={c['candidate_dispersion']:.4f} "
              f"null_disp={c['null_dispersion']:.4f} pos_edge={c['positive_edge']} "
              f"med={c['mean_candidate_median']:+.4f} null={c['mean_null_median']:+.4f}")