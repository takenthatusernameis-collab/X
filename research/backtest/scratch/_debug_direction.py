import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt

bars = bt.generate_bars(600, seed=19)
grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
stress = bt.stress_regime_scenarios()
up_down = [s for s in stress if s.name in ("strong_up", "strong_down")]
print("up_down scenarios:", [(s.name, s.regimes[0].drift_annual, s.regimes[0].vol_annual, s.seed) for s in up_down])
result = bt.regime_stress(
    bt.direction_signal, list(bars), grid,
    (("fast", 20.0), ("slow", 60.0)),
    scenarios=up_down,
    train_window=60, test_window=20, warmup=0,
)
print(result.inspect())
for s in result.scenarios:
    print(f"{s.name}: candidate={s.baseline_median_log_return:+.4f} noise={s.noise_median_log_return:+.4f} disp={abs(s.baseline_median_log_return) > bt.OUTLIER_TOL}")
print("candidate_disp=", result.candidate_dispersion, "null_disp=", result.null_dispersion)
