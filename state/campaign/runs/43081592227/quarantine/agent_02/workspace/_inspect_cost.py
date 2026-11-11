import json

with open("state/check_artifacts/momentum_cost_sensitivity_results.json") as f:
    art = json.load(f)

cl = art["cost_model_control"]["constant_long"]
tickers = list(cl["zero"].keys())
print("=== constant-long control: total return by cost level ===")
for t in tickers:
    z = cl["zero"][t]
    r = cl["realistic"][t]
    c = cl["conservative"][t]
    print("  {:6s} zero={:+.4f} real={:+.4f} cons={:+.4f} | net_sh={:+.0f} gross_sh={:,.0f} comm_z={:.0f} comm_r={:.0f} comm_c={:.0f}".format(
        t, z["const_long_total_return"], r["const_long_total_return"], c["const_long_total_return"],
        z["net_shares"], z["gross_shares"], z["commission"], r["commission"], c["commission"]))

print()
print("=== random AAPL control ===")
for lv in ["zero", "realistic", "conservative"]:
    print(" ", lv, art["cost_model_control"]["random_aapl"][lv])

print()
print("=== candidate mean segment median change (lookback 5) ===")
for t in art["summary"]["5"]:
    s = art["summary"]["5"][t]
    dz = s["realistic"]["mean_median"] - s["zero"]["mean_median"]
    dc = s["conservative"]["mean_median"] - s["zero"]["mean_median"]
    print("  {:6s} d_real={:+.4f} d_cons={:+.4f} class={}".format(t, dz, dc, s["cost_class"]))