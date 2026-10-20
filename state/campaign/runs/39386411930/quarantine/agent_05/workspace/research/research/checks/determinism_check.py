"""Determinism check for the asset-universe sweep (independent re-run)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from examples.ma_crossover import ma_crossover_signals


def key_fields():
    base_regimes = [
        bt.Regime(drift_annual=0.03, vol_annual=0.18, intraday_range_scale=0.015),
        bt.Regime(drift_annual=-0.01, vol_annual=0.35, intraday_range_scale=0.030),
    ]
    drift_offsets = [-0.06, -0.03, 0.00, 0.03, 0.06, 0.09, 0.12]
    assets = bt.uniform_regime_assets(
        base_regimes, n_assets=7, drift_offsets=drift_offsets, seed=42, n_bars=1000
    )
    grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
    result = bt.sweep_across_assets(
        ma_crossover_signals, assets, grid, (("fast", 20), ("slow", 60)),
        train_window=252, test_window=84, warmup=60, overlap_window=60,
    )
    summary = bt.asset_sweep_summary(
        result, (("fast", 20), ("slow", 60)), per_asset_null=True
    )
    return (
        summary.verdict,
        summary.median_log_returns,
        summary.asset_null_medians,
        summary.asset_significance,
        summary.n_significant_assets,
        summary.assets_positive_share,
        summary.null_dispersion,
    )


r1 = key_fields()
r2 = key_fields()
print("r1 == r2:", r1 == r2)
for name, a, b in zip(
    ("verdict", "medians", "null_medians", "significance",
     "n_sig", "best_share", "null_disp"), r1, r2):
    print(f"  {name}: {'identical' if a == b else 'DIFFERENT'}")
print("hash1:", hash(str(r1)))
print("hash2:", hash(str(r2)))
