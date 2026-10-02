import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import research.backtest as bt

bars = bt.generate_bars(600, seed=17)
grid = bt.parameter_grid_around((("fast", 20), ("slow", 60)))
scenarios = bt.stress_regime_scenarios()

result = bt.regime_stress(
    bt.always_long_signal, list(bars), grid,
    (("fast", 20.0), ("slow", 60.0)),
    scenarios=scenarios,
    train_window=60, test_window=20, warmup=0,
)
print(result.inspect())
for s in result.scenarios:
    print(f"{s.name}: candidate={s.baseline_median_log_return:+.4f} noise={s.noise_median_log_return:+.4f} edge={s.edge_status} status={'EDGE' if s.edge_status==bt.EdgePresent else 'noise'}")
print("spread_ratio=", result.spread_ratio)
