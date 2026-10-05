import json
p = "state/check_artifacts/momentum_sweep_results.json"
d = json.load(open(p))
print("top keys:", list(d.keys()))
print()
print("cross_summary:", json.dumps(d["cross_summary"], indent=1)[:1600])
print()
for lb in d["lookbacks"]:
    out = d["lookback_{}".format(lb)]
    print("=== lookback_{} ===".format(lb))
    print("  verdict_counts:", out["verdict_counts"])
    print("  n_regime_stable:", out["n_regime_stable"])
    for e in out["per_asset"]:
        print("    {:6s} medians={} null={} cd={:.4f} nd={:.4f} v={}".format(
            e["ticker"], e["medians"], e["null_medians"],
            e["candidate_dispersion"], e["null_dispersion"], e["verdict"]))
