import json

art = json.load(open("state/check_artifacts/momentum_cost_sensitivity_results.json"))
lb = "5"
print("lookback", lb)
for t in art["ticker_order"]:
    for m in ["0.0", "1.0", "2.0", "5.0"]:
        c = art["grid"][lb][m][t]
        cm = ",".join(f"{v:+.3f}" for v in c["medians"])
        nm = ",".join(f"{v:+.3f}" for v in c["null_medians"])
        print(f"{t:6} {m:>4} {c['verdict']:22} cand[{cm}] null[{nm}]")
    print()