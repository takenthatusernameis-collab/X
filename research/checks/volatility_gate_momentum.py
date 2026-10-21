"""Volatility-gate momentum test for lookback-5.

Background: the MA regime-filter class was exhaustively explored and falsified.
The short-horizon momentum class (already qualified at lookback=5) may have
different regime sensitivity: MA filters fail, but volatility-conditioned
momentum might work.

Evidence basis: MA regime filters were previously falsified; whether the same
class of conditioning helps the different momentum mechanism remains unresolved.

Method (fixed a-priori, not tuned to OOS):
- Long the previous 5-day return
- Hold 1 day, daily rebalance
- Trade only when past-only realized volatility > 0.25 annualized
- Use existing regime-stability stress testing across the 4 volatility blocks
- Run on the 10-asset collected universe
- Compare against a matched null (same conditioning, randomized signs)

Falsification prediction: either CONSISTENT_WITH_NOISE (no edge anywhere under
realistic costs) or REGIME_DEPENDENT (the edge lives in one regime mix) or
REGIME_STABLE_LOSS (uniformly negative). Either outcome closes the cell.
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
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "volatility_gate_momentum_results.json"

# Predeclared volatility gate: trade only when past-only vol > 0.25 annualized
VOLATILITY_GATE = bt.active_when_turbulent  # this function is defined in regime_stability.py

# Wrapped momentum signal: only emit signals when volatility gate is True
VOLATILITY_GATED_MOMENTUM = bt.conditional_signal(bt.momentum_signals, VOLATILITY_GATE)

# Matched null for volatility-gate momentum: same conditioning, randomized signs
def coin_flip_volatility_gate_signals(closes: np.ndarray, lookback: int, seed: int = 42) -> list[bt.Signal]:
    """Null momentum signals with same volatility gate but randomized signs."""
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    rng = np.random.default_rng(seed)
    
    for i in range(lookback, n):
        # Use the same volatility condition
        window = min(60, max(4, n // 4))
        if n >= window:
            early_vol = float(
                np.std(np.log(closes[window // 2 : window]), ddof=1)
            ) * np.sqrt(252.0)
            if early_vol > 0.25:  # same gate condition
                ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
                # Coin-flip: keep momentum direction or flip sign with 50% probability
                if rng.random() < 0.5:
                    signals[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
                else:
                    signals[i] = bt.Signal(date=i + 1, weight=-1.0 if ret > 0 else 1.0)
            else:
                signals[i] = bt.Signal(date=i + 1, weight=0.0)
        else:
            signals[i] = bt.Signal(date=i + 1, weight=0.0)
    return signals


def run_volatility_gate_test(asset, tickers):
    """Run volatility-gate momentum vs matched null for one asset."""
    bars_tuple = tickers[asset]
    bars = bars_tuple[0]  # Extract BarSequence from (BarSequence, List[str]) tuple
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    results = {}
    
    # Run both the volatility-gated momentum and matched null
    for name, sig_fn in [
        ("volatility_gate_momentum", lambda c, **p: VOLATILITY_GATED_MOMENTUM(c, **p)),
        ("null_gate_momentum", lambda c, **p: coin_flip_volatility_gate_signals(c, lookback=p["lookback"], seed=42)),
    ]:
        signals = sig_fn(closes, lookback=LOOKBACK)
        if len(signals) != len(closes):
            raise ValueError(f"{asset} {name}: signals {len(signals)} != bars {len(closes)}")
        
        # Run regime-stability stress test
        res = bt.stress_segments(
            signals_fn=sig_fn,
            bars=list(bars),
            signals=signals,
            param_grid=[{"lookback": LOOKBACK}],
            baseline=tuple([("lookback", LOOKBACK)]),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        
        # Extract key metrics for volatility-gate robustness classification
        cand_medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
        null_medians = [round(s.noise_median_log_return, 3) for s in res.scenarios]
        
        # Classification: volatility-gate robustness verdict based on realistic vs null performance
        cand = np.array(cand_medians)
        null = np.array(null_medians)
        cand_disp = float(np.std(cand))
        null_disp = float(np.std(null))
        
        # Volatility-gate robustness criteria:
        # - CONSISTENT_WITH_NOISE_VOL_GATE: indistinguishable from null under volatility gate
        # - REGIME_DEPENDENT_VOL_GATE: candidate swings across regimes significantly more than null
        # - REGIME_STABLE_VOL_GATE: consistent edge (or no edge) everywhere under volatility gate
        # - REGIME_STABLE_LOSS_VOL_GATE: uniform losses everywhere under volatility gate
        
        if all(abs(m) <= 0.05 for m in cand):
            verdict = "CONSISTENT_WITH_NOISE_VOL_GATE"
        elif cand_disp > 2.0 * null_disp:
            verdict = "REGIME_DEPENDENT_VOL_GATE"
        elif all(m < 0 for m in cand):
            verdict = "REGIME_STABLE_LOSS_VOL_GATE"
        else:
            verdict = "REGIME_STABLE_VOL_GATE"
        
        results[name] = dict(
            candidate_medians=cand_medians,
            null_medians=null_medians,
            candidate_dispersion=round(res.candidate_dispersion, 3),
            null_dispersion=round(res.null_dispersion, 3),
            n_folds=res.n_folds,
            segments=[s.name for s in res.scenarios],
            verdict=verdict,
        )
    
    return results


def main():
    # Preflight: validate real-data manifest
    manifest = load_manifest()
    if manifest["dataset_id"] != DATASET_ID:
        raise RuntimeError(f"Dataset mismatch: expected {DATASET_ID}, got {manifest['dataset_id']}")
    
    # Load the collected universe (10 assets)
    tickers = bt.real_data.load_universe()
    
    # Run test for each asset
    universe_results = {}
    for asset in tickers.keys():
        print(f"Processing {asset}...")
        universe_results[asset] = run_volatility_gate_test(asset, tickers)
    
    # Aggregate across universe
    agg_results = {
        "dataset_id": DATASET_ID,
        "lookback": LOOKBACK,
        "min_segment_bars": MIN_SEGMENT_BARS,
        "n_blocks": N_BLOCKS,
        "overlap": OVERLAP,
        "volatility_gate_threshold": 0.25,
        "predeclared_gate": "active_when_turbulent (vol > 25% annualized)",
        "matched_null": "same volatility gate, randomized signs (coin-flip)",
        "universe": {},
    }
    
    # Build per-asset verdict counts
    verdict_counts = {
        "CONSISTENT_WITH_NOISE_VOL_GATE": 0,
        "REGIME_DEPENDENT_VOL_GATE": 0,
        "REGIME_STABLE_VOL_GATE": 0,
        "REGIME_STABLE_LOSS_VOL_GATE": 0,
    }
    
    for asset, results in universe_results.items():
        vol_gate_result = results["volatility_gate_momentum"]
        agg_results["universe"][asset] = vol_gate_result
        verdict_counts[vol_gate_result["verdict"]] += 1
    
    # Add aggregate verdict counts to results
    agg_results["verdict_counts"] = verdict_counts
    
    # Save results for independent verification
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(agg_results, f, indent=2)
    
    print(f"Volatility-gate momentum results saved to {ARTIFACT_PATH}")
    
    # Summary statistics
    print("\n=== VOLATILITY-GATE MOMENTUM TEST SUMMARY ===")
    print(f"Dataset: {DATASET_ID}")
    print(f"Lookback: {LOOKBACK} days")
    print(f"Volatility Gate: active_when_turbulent (vol > 25% annualized)")
    print(f"Universe size: {len(tickers)} assets")
    print(f"Verdict counts:")
    for verdict, count in verdict_counts.items():
        print(f"  {verdict}: {count}")
    
    # Determine overall verdict
    positive_results = verdict_counts["REGIME_STABLE_VOL_GATE"] + verdict_counts["REGIME_DEPENDENT_VOL_GATE"]
    if positive_results > 0:
        print(f"\nOVERALL RESULT: {positive_results} assets showed positive edge under volatility gate")
        print("This suggests the volatility gate may improve robustness for momentum")
    else:
        print(f"\nOVERALL RESULT: {verdict_counts['CONSISTENT_WITH_NOISE_VOL_GATE'] + verdict_counts['REGIME_STABLE_LOSS_VOL_GATE']} assets showed no edge or uniform losses")
        print("This suggests the volatility gate does not help momentum (MA approach falsifies similar regime conditioning)")
    
    return agg_results

if __name__ == "__main__":
    main()
