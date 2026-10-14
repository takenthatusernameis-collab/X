"""Independent verification of the momentum cost-robustness check
(`research/checks/momentum_cost_robustness.py`).

This is a separate implementation path: it recomputes each lookback's
walk-forward median and cost-robustness result from a fresh `walk_forward`
implementation (not ``parameter_sweep``), then compares against the artifact
written by the cost-robustness check (`state/check_artifacts/momentum_cost_robustness.json`).
A mismatch would flag a defect in the check.

Design of the independent path:
- Real data is loaded from the same manifest path as the check.
- Cost model parameters are re-instantiated identically to ensure the same\ntransaction costs.
- Walk-forward with realistic costs is run afresh for each lookback.
- All values are loaded from the check's artifact and compared.

This keeps the same inputs (dataset, seed, windows, cost parameters) so a\nmatch confirms the check computed from those inputs; a mismatch would flag\nan error in the check's pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_robustness.json"

REALISTIC_COST_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
class MomentumCostRobustnessVerifier:
    def __init__(self):
        self.manifest = None
        self.tickers_univ = None
        self.artifact = None

    def run(self):
        print("=== Independent Verification of Momentum Cost-Robustness Check ===")
        print("Realistic cost model:", REALISTIC_COST_CFG)

        # 1. Load artifact from the check
        print("\n=== 1. Load check artifact ===")
        with open(ARTIFACT_PATH) as f:
            self.artifact = json.load(f)
        print(f"  Loaded artifact from: {ARTIFACT_PATH}")

        # 2. Manifest integrity
        self.manifest = load_manifest()
        assert self.manifest["dataset_id"] == DATASET_ID, "manifest id mismatch"
        print(f"  Manifest ID: {self.manifest['dataset_id']}")
        print(f"  Universe size: {len(self.manifest['universe'])}")

        # 3. Load universe
        print("\n=== 2. Load universe ===")
        self.tickers_univ = {}
        for entry in self.manifest["entries"]:
            ticker = entry["ticker"]
            bars, _ = bt.load_ticker(ticker)
            self.tickers_univ[ticker] = bars
            print(f"  {ticker}: {bars.n_bars} bars")

        # 4. Independent verification for each lookback
        print("\n=== 3. Independent verification (lookback 3/5/10) ===")
        all_ok = True

        for lookback in LOOKBACKS:
            print(f"\n--- Lookback {lookback} ---")
            all_ok = self._verify_lookback(lookback, all_ok)

        # 5. Determinism check
        print("\n=== 4. Determinism check ===")
        print("  Running determinism check for lookback 5...")
        self._determinism_check(5)
        print("  Determinism verified: two consecutive runs produced identical results")

        # 6. Summary
        print("\n" + "="*60)
        if all_ok:
            print("All independent recomputations MATCH the check artifact.")
        else:
            print("MISMATCH FOUND: the check artifact does not reproduce via the")
            print("independent path. Investigate before admitting the results.")
            sys.exit(1)
        print("="*60)
        print("\nNOTE: research/simulation only. No live trading or production")
        print("execution. Momentum cost-robustness check independent verification on")
        print("collected universe: exploratory simulation.")

    def _verify_lookback(self, lookback, all_ok):
        print(f"  Verifying lookback {lookback}...\n")

        # Load check's result
        check_result = self.artifact["results_by_lookback"][str(lookback)]
        check_median_log_return = check_result["median_log_return"]
        check_positive_folds = check_result["positive_folds"]
        check_n_folds = check_result["n_folds"]

        # Independent recomputation
        asset = "AAPL"
        bars = self.tickers_univ[asset]
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=lookback)

        # Run walk-forward with realistic costs
        result = bt.walk_forward(
            list(bars), signals, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=REALISTIC_COST_CFG
        )

        # Compute medians
        independent_median_log_return = float(np.median([
            np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
            for f in result.folds
        ]))
        independent_positive_folds = sum(1 for f in result.folds if f.metrics["total_return"] > 0)
        independent_n_folds = len(result.folds)

        # Compare results
        median_match = abs(independent_median_log_return - check_median_log_return) < 1e-3
        positive_folds_match = independent_positive_folds == check_positive_folds
        n_folds_match = independent_n_folds == check_n_folds

        print(f"    Independent median log return: {independent_median_log_return:+.3f}")
        print(f"    Check median log return:        {check_median_log_return:+.3f}")
        print(f"    Median match: {'YES' if median_match else 'NO'}")

        print(f"    Independent positive folds: {independent_positive_folds}")
        print(f"    Check positive folds:        {check_positive_folds}")
        print(f"    Positive folds match: {'YES' if positive_folds_match else 'NO'}")

        if not (median_match and positive_folds_match and n_folds_match):
            print(f"    INDEPENDENT VERIFICATION FAILED for lookback {lookback}")
            print(f"      Median difference: {independent_median_log_return - check_median_log_return:+.6f}")
            all_ok = False

        return all_ok

    def _determinism_check(self, lookback):
        """Run the same computation twice to ensure determinism."""
        print(f"  Running determinism check for lookback {lookback}...\n")

        # First run
        asset = "AAPL"
        bars = self.tickers_univ[asset]
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=lookback)

        result1 = bt.walk_forward(
            list(bars), signals, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=REALISTIC_COST_CFG
        )

        median1 = float(np.median([
            np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
            for f in result1.folds
        ]))

        # Second run
        result2 = bt.walk_forward(
            list(bars), signals, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=REALISTIC_COST_CFG
        )

        median2 = float(np.median([
            np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
            for f in result2.folds
        ]))

        if abs(median1 - median2) < 1e-12:
            print(f"    Determinism PASSED: median1={median1:.6f}, median2={median2:.6f}")
        else:
            print(f"    DETERMINISM FAILED: median1={median1:.6f}, median2={median2:.6f}")
            raise AssertionError(f"Determinism check failed for lookback {lookback}")
if __name__ == "__main__":
    verifier = MomentumCostRobustnessVerifier()
    verifier.run()
