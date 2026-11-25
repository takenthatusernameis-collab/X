"""Summarize the cost-sensitivity artifact: per-asset medians across cost levels."""
import json
from pathlib import Path

art = json.loads(Path("state/check_artifacts/momentum_cost_sensitivity_results.json").read_text())
levels = art["cost_order"]
for lb in ["3", "5", "10"]:
    print(f"=== lookback {lb} ===")
    for t in art["universe"]["ticker_order"]:
        pa = art["per_asset"][t][lb]
        meds = {lvl: pa[lvl]["medians"] for lvl in levels}
        verdicts = {lvl: pa[lvl]["verdict"] for lvl in levels}
        # max abs change in any segment median vs zero
        z = meds["zero"]
        maxchg = {lvl: round(max(abs(a - b) for a, b in zip(meds[lvl], z)), 4) for lvl in levels}
        print(f"  {t:6s} zero={z}")
        for lvl in levels:
            print(f"         {lvl:7s} med={meds[lvl]} maxchg_vs_zero={maxchg[lvl]:.4f} verdict={verdicts[lvl]}")
    print()