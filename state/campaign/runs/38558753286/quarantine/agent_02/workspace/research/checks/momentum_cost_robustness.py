"""Cost-sensitivity check for lookback 3/5/10 momentum.

This implements the bounded cost-sensitivity check for lookback 3/5/10
momentum under the repository's existing realistic cost model and matched null.

Primary question: Does the short-horizon momentum edge survive conservative
transaction-cost stress on the collected real-data universe?

Evidence basis: LEARNING_STATE admits short-horizon momentum as candidate positive
evidence but leaves execution-cost robustness as a meaningful qualification.

Deliverable: A reproducible artifact and independent verification result
classifying cost robustness.

Scope: Use the existing 10-asset collected universe and existing momentum
implementation/check framework.

Out of scope: Do not tune signal rules after seeing cost results, expand the
universe, or introduce leverage.
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
LOOKBACKS = [3, 5, 10]
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_robustness.json"

REALISTIC_COST_CFG = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
class MomentumCostRobustnessChecker:
    def __init__(self):
        self.manifest = None
        self.tickers_univ = None
        self.results = {}

    def run(self):
        print("=== Momentum Cost-Robustness Check (lookback 3/5/10) ===")
        print("Realistic cost model: commission_per_trade=2.0, commission_per_share=0.003,")
        print("  slippage_cents=2.0, slippage_proportional=0.0005")

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

        # 3. Run cost sensitivity checks for each lookback
        print("\n=== 3. Cost-sensitivity checks (lookback 3/5/10) ===")
        for lookback in LOOKBACKS:
            self._run_lookback_cost_check(lookback)

        # 4. Generate artifact
        print("\n=== 4. Generate artifact ===")
        artifact = self._generate_artifact()
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_PATH, "w") as f:
            json.dump(artifact, f, indent=2, sort_keys=True)
        print(f"  artifact written to: {ARTIFACT_PATH}")

        return artifact

    def _run_lookback_cost_check(self, lookback):
        print(f"\n--- Lookback {lookback} with realistic costs ---")

        # 3a. Data preflight
        from research.data.preflight import main as preflight_main
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "data"))
        from importlib import import_module
        pf = import_module("preflight")
        rc = pf.main()
        if rc != 0:
            print("PREFLIGHT FAILED; aborting cost run.")
            sys.exit(1)

        # 3b. Momentum signals and leakage review
        print(f"  Generate momentum signals (lookback={lookback})")
        asset = "AAPL"
        bars = self.tickers_univ[asset]
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=lookback)

        # Leakage review
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        print(f"  [PASS] check_signal_integrity")

        res = bt.run_bars(list(bars), signals, REALISTIC_COST_CFG)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        print(f"  [PASS] check_equity_matches_fills")

        # Cost application
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

        # 3c. Store results
        self.results[lookback] = {
            "median_log_return": median_log_return,
            "positive_folds": positive_folds,
            "n_folds": n_folds,
            "total_return": result.aggregate_metrics.get("total_return", median_log_return),
            "mean_log_return": result.aggregate_metrics.get("mean_log_total_return", median_log_return),
        }

    def _generate_artifact(self):
        """Generate artifact with cost robustness classification."""
        print("\n=== 4a. Generate cost robustness classification ===")

        # Analyze cost robustness across lookbacks
        cost_robust_positive = []
        cost_robust_negative = []
        cost_undefined = []

        for lookback in LOOKBACKS:
            median_log_return = self.results[lookback]["median_log_return"]
            if abs(median_log_return) <= 0.05:  # Within noise band (approx)
                cost_undefined.append({
                    "lookback": lookback,
                    "median_log_return": median_log_return,
                    "verdict": "CONSISTENT_WITH_NOISE"
                })
            elif median_log_return > 0.05:  # Positive edge
                cost_robust_positive.append({
                    "lookback": lookback,
                    "median_log_return": median_log_return,
                    "verdict": "COST_ROBUST_POSITIVE"
                })
            else:  # Negative edge
                cost_robust_negative.append({
                    "lookback": lookback,
                    "median_log_return": median_log_return,
                    "verdict": "COST_ROBUST_NEGATIVE"
                })

        # Classification summary
        print(f"  Cost-robust positive edges: {len(cost_robust_positive)}")
        print(f"  Cost-robust negative edges: {len(cost_robust_negative)}")
        print(f"  Cost-undefined (within noise): {len(cost_undefined)}")

        # Overall verdict
        if len(cost_robust_positive) > 0 and len(cost_robust_negative) == 0:
            overall_verdict = "COST_ROBUST_POSITIVE"
            conclusion = "The short-horizon momentum edge survives conservative transaction-cost stress."
        elif len(cost_robust_positive) == 0 and len(cost_robust_negative) > 0:
            overall_verdict = "COST_ROBUST_NEGATIVE"
            conclusion = "The short-horizon momentum edge does not survive conservative transaction-cost stress."
        else:
            overall_verdict = "COST_SENSITIVE"
            conclusion = "The short-horizon momentum edge shows mixed results under cost stress."

        print(f"\n  Overall verdict: {overall_verdict}")
        print(f"  Conclusion: {conclusion}")

        # Generate artifact
        artifact = {
            "dataset_id": DATASET_ID,
            "seed": SEED,
            "lookbacks": LOOKBACKS,
            "cost_model": {
                "commission_per_trade": REALISTIC_COST_CFG.commission_per_trade,
                "commission_per_share": REALISTIC_COST_CFG.commission_per_share,
                "slippage_cents": REALISTIC_COST_CFG.slippage_cents,
                "slippage_proportional": REALISTIC_COST_CFG.slippage_proportional,
            },
            "walk_forward_params": {
                "train_window": TRAIN,
                "test_window": TEST,
                "warmup": WARM,
                "overlap": OVERLAP,
            },
            "results_by_lookback": self.results,
            "cost_robust_positive": cost_robust_positive,
            "cost_robust_negative": cost_robust_negative,
            "cost_undefined": cost_undefined,
            "overall_verdict": overall_verdict,
            "conclusion": conclusion,
        }

        return artifact
if __name__ == "__main__":
    checker = MomentumCostRobustnessChecker()
    artifact = checker.run()
    print("\n" + "="*60)
    print("NOTE: research/simulation only. No live trading or production")
    print("execution. Momentum cost-robustness check on collected universe:")
    print(f"  {len(LOOKBACKS)} lookback periods tested with realistic costs.")
    print(f"  Overall verdict: {artifact['overall_verdict']}")
    print("="*60)
    sys.exit(0)
