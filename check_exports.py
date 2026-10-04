import sys
sys.path.insert(0, "/home/runner/work/X/X")
import research.backtest as bt
checks = {
    "stress_segments": hasattr(bt, "stress_segments"),
    "segment_fn_from_labels": hasattr(bt, "segment_fn_from_labels"),
    "volatility_segments": hasattr(bt, "volatility_segments"),
    "SweepSummary.noise_fold_median_log_returns": hasattr(bt.SweepSummary, "noise_fold_median_log_returns"),
    "SweepSummary.effective_tolerance": hasattr(bt.SweepSummary, "effective_tolerance"),
    "compare_noise": hasattr(bt.SweepSummary, "compare_noise"),
}
for k, v in checks.items():
    print(f"{k}: {v}")
assert all(checks.values()), "some exports missing"
print("ALL_EXPORTS_OK")
