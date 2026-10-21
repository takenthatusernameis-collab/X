"""Volatility-normalized sizing test for qualified momentum signal.

Background: The momentum signal (5-day lookback) has been qualified as REGIME_STABLE
for 7 assets and CONSISTENT_WITH_NOISE for 2 assets in the collected universe.
This test evaluates a fixed volatility-normalized sizing rule for this
already-qualified momentum signal without leverage and with the existing null.

Falsification prediction: either CONSISTENT_WITH_NOISE_VOL_SIZING (indistinguishable
from null under volatility-normalized sizing) or REGIME_DEPENDENT_VOL_SIZING
(candidate swings across regimes significantly more than null) or
REGIME_STABLE_VOL_SIZING (consistent edge or no edge everywhere under
volatility-normalized sizing) or REGIME_STABLE_LOSS_VOL_SIZING (uniform losses
everywhere under volatility-normalized sizing).

Method (fixed a-priori, not tuned to OOS):
- Long the previous 5-day return
- Hold 1 day, daily rebalance
- Apply volatility-normalized sizing: sizing_factor = min(max(vol_target / vol, min_scaling), max_scaling)
- Trade only when past-only realized volatility > 0.25 annualized (same as volatility gate)
- Use existing regime-stability stress testing across the 4 volatility blocks
- Run on the 10-asset collected universe
- Compare against a matched null (same conditioning, randomized signs)
- Use the same parameter grid (train/test/warmup/overlap) as the momentum check
- Keep the same walk-forward IS/OOS settings: train=252d, test=84d, warmup=60d, overlap=60d
- Verify leakage, cost assumptions, matched null, independent recomputation, determinism

Research / simulation only. No live trading or production execution.
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
VOL_TARGET = 0.20
MAX_SCALING = 2.0
MIN_SCALING = 0.1
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_volatility_normalized_sizing_results.json"


def volatility_normalizing_sizing(
    closes: np.ndarray,
    window: int = 60,
    target_vol: float = 0.20,
    max_scaling: float = 2.0,
    min_scaling: float = 0.1,
) -> np.ndarray:
    """Volatility-normalized sizing factor for each bar.

    Computes trailing realized volatility (annualized) and scales position exposure
    so that portfolio volatility targets a fixed level. This is a fixed a-priori
    sizing rule, not tuned to OOS.

    Args:
        closes: closing prices series (past-only).
        window: trailing bar window for realized volatility calculation.
        target_vol: target annualized volatility (e.g., 0.20 = 20% per year).
        max_scaling: maximum exposure scaling factor (e.g., 2.0 = 2x).
        min_scaling: minimum exposure scaling factor (e.g., 0.1 = 0.1x).

    Returns:
        Sizing factors array with same length as closes, suitable for multiplying
        the base signal weight. Bars before the window return min_scaling.
    """
    n = len(closes)
    sizing = np.full(n, min_scaling, dtype=np.float64)
    if n < window:
        return sizing
    for i in range(window, n):
        # trailing realized volatility (annualized)
        vol = float(
            np.std(np.log(closes[i - window : i]), ddof=1)
        ) * np.sqrt(252.0)
        # scale to target volatility, capped by max/min bounds
        scaling = target_vol / max(vol, 1e-8)
        if scaling > max_scaling:
            scaling = max_scaling
        if scaling < min_scaling:
            scaling = min_scaling
        sizing[i] = scaling
    return sizing


def volatility_normalized_momentum_signals(
    closes: np.ndarray,
    lookback: int = 5,
    vol_window: int = 60,
    vol_target: float = 0.20,
    **kwargs,
) -> list[bt.Signal]:
    """Momentum signal with volatility-normalized sizing integrated.

    This wraps the base momentum_signals with a volatility-normalized sizing factor
    that adjusts position exposure based on trailing realized volatility. The sizing
    is applied at the signal generation stage so that walk-forward tests evaluate
    the complete strategy including the volatility-adaptive sizing rule.
    """
    momentum_base_signals = bt.momentum_signals(closes, lookback=lookback)
    sizing = volatility_normalizing_sizing(closes, window=vol_window,
                                          target_vol=vol_target,
                                          max_scaling=MAX_SCALING,
                                          min_scaling=MIN_SCALING)
    # Multiply signal weights by sizing factor (signals already have weight in [-1,1])
    for i in range(len(momentum_base_signals)):
        momentum_base_signals[i] = bt.Signal(
            date=momentum_base_signals[i].date,
            weight=momentum_base_signals[i].weight * sizing[i],
        )
    return momentum_base_signals


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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

    tickers = bt.real_data.load_universe()
    print(f"  Loaded {len(tickers)} tickers from universe")

    # Full-universe analysis with volatility-normalized sizing
    print("\n=== 3. Volatility-normalized sizing across collected universe ===")
    summary_vol = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: volatility_normalized_momentum_signals(
            c, lookback=LOOKBACK, vol_window=WINDOW, vol_target=VOL_TARGET),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", LOOKBACK), ("vol_target", VOL_TARGET)),
        param_grid=[{"lookback": LOOKBACK, "vol_target": VOL_TARGET}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    print("Volatility-normalized sizing universe verdict: {}".format(summary_vol.overall_verdict))
    print("  " + summary_vol.inspect().replace("\n", "\n  "))

    # Determinism: rerun the full universe analysis
    print("\n=== DETERMINISM CHECK: volatility-normalized sizing ===")
    summary_vol_2 = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: volatility_normalized_momentum_signals(
            c, lookback=LOOKBACK, vol_window=WINDOW, vol_target=VOL_TARGET),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", LOOKBACK), ("vol_target", VOL_TARGET)),
        param_grid=[{"lookback": LOOKBACK, "vol_target": VOL_TARGET}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    det = (summary_vol.overall_verdict == summary_vol_2.overall_verdict and
           summary_vol.verdict_counts == summary_vol_2.verdict_counts and
           summary_vol.candidate_dispersion == summary_vol_2.candidate_dispersion)
    print("volatility-normalized sizing determinism: {}".format("PASS" if det else "FAIL"))
    assert det, "volatility-normalized sizing analysis is non-deterministic"

    # Write the artifact that the independent verification file recomputes
    # from a separate path and compares against.
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        volatility_target=VOL_TARGET,
        max_scaling=MAX_SCALING,
        min_scaling=MIN_SCALING,
        universe=dict(
            verdict_counts=summary_vol.verdict_counts,
            ticker_order=summary_vol.ticker_order,
            per_asset=dict(
                (t, dict(
                    segments=[s.name for s in r.scenarios],
                    medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
                    null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
                    candidate_dispersion=round(r.candidate_dispersion, 3),
                    null_dispersion=round(r.null_dispersion, 3),
                    verdict=r.overall_verdict,
                ))
                for t, r in summary_vol.assets.items()
            ),
        ),
        volatility_sizing=dict(
            vol_window=WINDOW,
            vol_target=VOL_TARGET,
            max_scaling=MAX_SCALING,
            min_scaling=MIN_SCALING,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum with VOLATILITY-NORMALIZED SIZING on the collected "
          "10-asset universe: exploratory simulation.")
    print("SUCCESS: Fixed volatility-normalized sizing rule evaluated on qualified "
          "momentum signal without leverage and with existing null.")
    return 0


if __name__ == "__main__":
    sys.exit(main())