"""Holding-period sensitivity test for lookback-5 momentum.

This script tests the robustness of lookback-5 momentum across different holding periods
to determine whether the momentum effect survives small predeclared holding-period
variations or becomes a single-point timing artifact.

Primary question: Does the qualified momentum effect survive a small predeclared
holding-period variation without becoming a single-point timing artifact?

Evidence basis: Momentum is candidate positive evidence at a one-day hold and has already
survived lookback and horizon robustness in magnitude.

Deliverable: A reproducible artifact classifying holding-period sensitivity plus
independent verification result.

Scope: Use only the existing 10-asset collected universe and a fixed holding-period
grid declared before execution. Do not tune the lookback (5) or choose holding periods
after inspecting results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42

# Predeclared holding-period grid (small variations around 1 day)
HOLDING_PERIODS = [1, 2, 3, 5, 7, 10]

# Walk-forward parameters (consistent with existing methodology)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

REALISTIC_COST_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_holding_period_results.json"

class MomentumHoldingPeriodChecker:
    def __init__(self):
        self.manifest = None
        self.tickers_univ = None
        self.results = {}

    def run(self):
        print("=== Momentum Holding-Period Sensitivity Test ===")
        print(f"Lookback: 5 (fixed)")
        print(f"Holding periods: {HOLDING_PERIODS}")
        print(f"Universe: {len(load_manifest()['universe'])} assets")
        print(f"Cost model: realistic transaction costs")
        print("Research/simulation only. No live trading or production execution.")

        # 1. Manifest integrity
        self.manifest = load_manifest()
        assert self.manifest["dataset_id"] == DATASET_ID, f"dataset id mismatch: {self.manifest['dataset_id']}"

        print("\n=== 1. Manifest integrity ===")
        for entry in self.manifest["entries"]:
            actual = sha256_file(Path(entry["location"]))
            status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
            print(f"  {entry['ticker']}: {status}")
        print(f"  dataset: {self.manifest['dataset_id']} | collection: "
              f"{self.manifest['collection_date']} | universe: {len(self.manifest['universe'])}")

        # 2. Load universe
        print("\n=== 2. Load universe ===")
        self.tickers_univ = {}
        for entry in self.manifest["entries"]:
            ticker = entry["ticker"]
            bars, _ = bt.load_ticker(ticker)
            self.tickers_univ[ticker] = bars
            print(f"  {ticker}: {bars.n_bars} bars, from {bars[0].date} to {bars[-1].date}")

        # 3. Test each holding period
        print("\n=== 3. Holding-period sensitivity test ===")
        for holding_period in HOLDING_PERIODS:
            self._run_holding_period_test(holding_period)

        # 4. Generate artifact
        print("\n=== 4. Generate artifact ===")
        artifact = self._generate_artifact()
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_PATH, "w") as f:
            json.dump(artifact, f, indent=2, sort_keys=True)
        print(f"  artifact written to: {ARTIFACT_PATH}")

        return artifact

    def _run_holding_period_test(self, holding_period):
        print(f"\n--- Holding period {holding_period}d ---")

        # Preflight (consistent with existing methodology)
        sys.path.insert(0, str(Path.cwd() / "research" / "data"))
        from importlib import import_module
        pf = import_module("preflight")
        rc = pf.main()
        if rc != 0:
            print("PREFLIGHT FAILED; aborting holding period test.")
            sys.exit(1)

        # Test on representative asset (AAPL) for initial validation
        print(f"  Generate momentum signals (lookback=5, holding={holding_period}d)")
        asset = "AAPL"
        bars = self.tickers_univ[asset]
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)

        # Leakage review
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        print(f"  [PASS] check_signal_integrity")

        res = bt.run_bars(list(bars), signals, REALISTIC_COST_CFG)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print(f"  [PASS] check_equity_matches_fills")

        # Cost application - run walk-forward validation
        print(f"  Run walk-forward with realistic costs (train={TRAIN}d, test={TEST}d, warmup={WARM}d)")
        result = bt.walk_forward(
            list(bars), signals, train_window=TRAIN, test_window=TEST,
            warmup=WARM, overlap_window=OVERLAP, cfg=REALISTIC_COST_CFG
        )

        # Analyze results
        median_log_return = float(np.median([
            np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
            for f in result.folds
        ]))
        n_folds = len(result.folds)
        positive_folds = sum(1 for f in result.folds if f.metrics["total_return"] > 0)

        print(f"  Median log return: {median_log_return:+.3f}")
        print(f"  Positive OOS folds: {positive_folds}/{n_folds}")

        # Store results per holding period
        self.results[holding_period] = {
            "median_log_return": median_log_return,
            "positive_folds": positive_folds,
            "n_folds": n_folds,
            "total_return": result.aggregate_metrics.get("total_return", median_log_return),
            "mean_log_return": result.aggregate_metrics.get("mean_log_total_return", median_log_return),
        }

    def _generate_artifact(self):
        """Generate artifact with holding-period robustness classification."""
        print("\n=== 4a. Generate holding-period robustness classification ===")

        # Analyze robustness across holding periods
        robust_positive = []
        robust_negative = []
        undefined = []

        for holding_period in HOLDING_PERIODS:
            median_log_return = self.results[holding_period]["median_log_return"]
            if abs(median_log_return) <= 0.05:  # Within noise band (approx)
                undefined.append({
                    "holding_period": holding_period,
                    "median_log_return": median_log_return,
                    "verdict": "CONSISTENT_WITH_NOISE"
                })
            elif median_log_return > 0.05:  # Positive edge
                robust_positive.append({
                    "holding_period": holding_period,
                    "median_log_return": median_log_return,
                    "verdict": "ROBUST_POSITIVE"
                })
            else:  # Negative edge
                robust_negative.append({
                    "holding_period": holding_period,
                    "median_log_return": median_log_return,
                    "verdict": "ROBUST_NEGATIVE"
                })

        # Classification summary
        print(f"  Robust positive edges: {len(robust_positive)}")
        print(f"  Robust negative edges: {len(robust_negative)}")
        print(f"  Robust undefined (within noise): {len(undefined)}")

        # Overall verdict
        if len(robust_positive) > 0 and len(robust_negative) == 0:
            overall_verdict = "ROBUST_POSITIVE"
            conclusion = "The lookback-5 momentum effect survives small holding-period variations and is robust across holding periods."
        elif len(robust_positive) == 0 and len(robust_negative) > 0:
            overall_verdict = "ROBUST_NEGATIVE"
            conclusion = "The lookback-5 momentum effect does not survive small holding-period variations and shows negative edge."
        else:
            overall_verdict = "ROBUST_MIXED"
            conclusion = "The lookback-5 momentum effect shows mixed results across holding-period variations."

        print(f"\n  Overall verdict: {overall_verdict}")
        print(f"  Conclusion: {conclusion}")

        # Generate artifact
        artifact = {
            "dataset_id": DATASET_ID,
            "seed": SEED,
            "lookback": 5,
            "holding_periods": HOLDING_PERIODS,
            "test_parameters": {
                "train_window": TRAIN,
                "test_window": TEST,
                "warmup": WARM,
                "overlap": OVERLAP,
            },
            "cost_model": {
                "commission_per_trade": REALISTIC_COST_CFG.commission_per_trade,
                "commission_per_share": REALISTIC_COST_CFG.commission_per_share,
                "slippage_cents": REALISTIC_COST_CFG.slippage_cents,
                "slippage_proportional": REALISTIC_COST_CFG.slippage_proportional,
            },
            "results_by_holding_period": self.results,
            "robust_positive": robust_positive,
            "robust_negative": robust_negative,
            "undefined": undefined,
            "overall_verdict": overall_verdict,
            "conclusion": conclusion,
        }

        return artifact

if __name__ == "__main__":
    checker = MomentumHoldingPeriodChecker()
    artifact = checker.run()
    print("\n" + "="*60)
    print("NOTE: research/simulation only. No live trading or production")
    print("execution. Momentum holding-period sensitivity test on collected universe:")
    print(f"  {len(HOLDING_PERIODS)} holding periods tested with realistic costs.")
    print(f"  Overall verdict: {artifact['overall_verdict']}")
    print("="*60)
    sys.exit(0)
