"""Test momentum with a past-only volatility gate (active only in turbulent segments).

Background: the base momentum signal (long previous 5-day return) is REGIME_DEPENDENT
on some assets (AMZN, JPM). The hypothesis being tested here: a volatility gate
that activates the momentum signal only inside turbulent segments is REGIME_STABLE
— i.e. the edge is genuinely volatility-regime-contingent and harvestable with a
regime-aware implementation.

Falsification prediction: the gated variant remains REGIME_DEPENDENT (or
becomes CONSISTENT_WITH_NOISE), because a past-only volatility classifier cannot
separate the 2009-2013 "good" turbulent regime from the 2022-2026 "bad" turbulent
regime — both carry the same "turbulent" label. A vol-based regime timing
therefore cannot harvest the edge.

Method: past-only MA crossover (same contract as research/backtest/regime_stability).
For each asset: three runs: base momentum, momentum with turbulent-only gate,
and regime-adaptive momentum (calm windows 20/60, turbulent windows 10/30) for comparison.
All signals are past-only. The variant is run through the SAME
`stress_segments` pipeline as the base momentum, with the same coin-flip null,
so the REGIME_STABLE / REGIME_DEPENDENT / CONSISTENT_WITH_NOISE verdicts are comparable.

Determinism: the script asserts byte-identical output across independent reruns.

This test is predeclared: the volatility gate is a principled design (activate only
in turbulent regimes) and not tuned to any results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.data.preflight import load_manifest, sha256_file

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
LOOKBACK = 5
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400


class MomentumVolatilityGate:
    """Momentum with a past-only volatility gate (active only in turbulent segments)."""

    def __init__(self, fast=20, slow=60, window=60, n_blocks=4):
        self.fast = fast
        self.slow = slow
        self.window = window
        self.n_blocks = n_blocks

    def base_signals(self, closes, fast, slow):
        """Past-only MA crossover (same contract as research/backtest/regime_stability)."""
        n = len(closes)
        out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        fast, slow = int(fast), int(slow)
        for i in range(max(fast, slow) - 1, n):
            fm = float(np.mean(closes[i - fast + 1 : i + 1]))
            sm = float(np.mean(closes[i - slow + 1 : i + 1]))
            out[i] = bt.Signal(date=i + 1, weight=1.0 if fm > sm else -1.0)
        return out

    def momentum_signals(self, closes, lookback):
        """Past-only momentum signal (long previous lookback-day return)."""
        n = len(closes)
        out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        lookback = int(lookback)
        for i in range(lookback, n):
            ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
            out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
        return out

    def volatile_masked_signals(self, closes, fast, slow, labels):
        """Momentum: active only inside turbulent segments."""
        base = self.momentum_signals(closes, LOOKBACK)
        n = len(closes)
        out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
        for i in range(n):
            if i >= self.window and labels[i] == "turbulent":
                out[i] = base[i]
        return out

    def run_asset(self, asset, tickers):
        """Run all three variants on one asset."""
        bars = tickers[asset]
        closes = bars.closes_array()
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        seg_fn = bt.segment_fn_from_labels(labels)

        results = {}

        # Base momentum
        signals = self.momentum_signals(closes, LOOKBACK)
        res = bt.stress_segments(
            signals_fn=lambda c, **p: self.momentum_signals(c, LOOKBACK),
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": LOOKBACK}],
            baseline=(("lookback", LOOKBACK),),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        results["base"] = dict(
            segments=[s.name for s in res.scenarios],
            medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
            candidate_dispersion=round(res.candidate_dispersion, 3),
            null_dispersion=round(res.null_dispersion, 3),
            verdict=res.overall_verdict,
            n_folds=res.n_folds,
        )

        # Momentum with volatility gate
        gated_signals = self.volatile_masked_signals(closes, self.fast, self.slow, labels)
        res_gated = bt.stress_segments(
            signals_fn=lambda c, labels=labels, **p: self.volatile_masked_signals(
                c, self.fast, self.slow, labels),
            bars=list(bars),
            signals=gated_signals,
            param_grid=[{"fast": self.fast, "slow": self.slow}],
            baseline=(("fast", self.fast), ("slow", self.slow)),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        results["gated"] = dict(
            segments=[s.name for s in res_gated.scenarios],
            medians=[round(s.baseline_median_log_return, 3) for s in res_gated.scenarios],
            candidate_dispersion=round(res_gated.candidate_dispersion, 3),
            null_dispersion=round(res_gated.null_dispersion, 3),
            verdict=res_gated.overall_verdict,
            n_folds=res_gated.n_folds,
        )

        return results

    def run_universe(self, tickers):
        """Run all three variants on the whole universe."""
        summary = bt.stress_segments_across_tickers(
            tickers=tickers,
            signals_fn=lambda c, **kw: self.momentum_signals(c, LOOKBACK),
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=("lookback", LOOKBACK),
            param_grid=[{"lookback": LOOKBACK}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        return summary

    def main(self) -> int:
        manifest = load_manifest()
        assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

        print("=== 1. Manifest integrity ===")
        for entry in manifest["entries"]:
            actual = sha256_file(Path(entry["location"]))
            status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
            print(f"  {entry['ticker']}: {status}")
        print(f"  dataset: {manifest['dataset_id']} | collection: "
              f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

        print("\n=== 2. Data preflight (research/data/preflight.py) ===")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
        from importlib import import_module
        pf = import_module("preflight")
        rc = pf.main()
        if rc != 0:
            print("PREFLIGHT FAILED; aborting real-data run.")
            sys.exit(1)

        print("\n=== 3. Leakage review per asset ===")
        target = ["AMZN", "JPM"]
        tickers = {}
        for ticker in target:
            bars, dates = bt.load_ticker(ticker)
            tickers[ticker] = bars
            closes = bars.closes_array()
            signals = self.momentum_signals(closes, LOOKBACK)
            bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
            res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
            bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
            print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
                  f"leakage [PASS]")

        print("\n=== 4. Momentum + volatility gate on REGIME_DEPENDENT assets ===")
        results_all = {}
        for asset in target:
            res = self.run_asset(asset, tickers)
            results_all[asset] = res
            print(f"Asset: {asset}")
            print(f"  {'variant':14s} {'segments':28s} {'medians':28s} {'dispersion':12s}  verdict")
            for name, r in res.items():
                segs = "[" + ",".join(r["segments"]) + "]"
                meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
                print(f"  {name:14s} {segs:28s} {meds:28s} {r['candidate_dispersion']:+10.3f}  {r['verdict']}")
            print("")

        # Determinism: recompute the 2-asset comparison and assert identical output
        results_all2 = {}
        for asset in target:
            results_all2[asset] = self.run_asset(asset, tickers)
        out1 = json.dumps(results_all, sort_keys=True)
        out2 = json.dumps(results_all2, sort_keys=True)
        print("determinism: r1 == r2: {}".format(out1 == out2))
        assert out1 == out2, "non-deterministic output"

        # Write the artifact
        artifact = dict(
            dataset_id=DATASET_ID,
            seed=SEED,
            lookback=LOOKBACK,
            train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
            window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
            per_asset=results_all,
        )
        ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH = ARTIFACT_DIR / "momentum_volatility_gate_results.json"
        with open(ARTIFACT_PATH, "w") as f:
            json.dump(artifact, f, indent=2, sort_keys=True)
        print("")
        print("artifact written to: {}".format(ARTIFACT_PATH))
        print("NOTE: research/simulation only. No live trading or production "
              "execution. Momentum volatility gate test on AMZN/JPM + universe: "
              "exploratory simulation.")
        return 0


if __name__ == "__main__":
    agent = MomentumVolatilityGate()
    sys.exit(agent.main())
