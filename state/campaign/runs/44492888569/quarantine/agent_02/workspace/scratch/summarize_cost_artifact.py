"""Compact summary of the momentum cost-sensitivity artifact."""
import json

PATH = "state/check_artifacts/momentum_cost_sensitivity_results.json"

with open(PATH) as fh:
    art = json.load(fh)

for ticker, per_lb in art["per_asset"].items():
    for lb_key, rec in per_lb.items():
        if rec["classification"] == "COST_ROBUST" and ticker not in ("AAPL", "MSFT"):
            continue
        print(f"\n{ticker} {lb_key} {rec['classification']} break_even={rec['break_even_multiplier']}")
        for mult in ("0.0", "0.5", "1.0", "2.0"):
            c = rec["by_cost"][mult]
            print(f"  {mult:>3s}x cand={c['medians']} null={c['null_medians']} "
                  f"disp={c['candidate_dispersion']}/{c['null_dispersion']} "
                  f"verdict={c['verdict']}")