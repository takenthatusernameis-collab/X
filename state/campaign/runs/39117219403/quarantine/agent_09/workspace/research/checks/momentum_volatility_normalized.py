"""Test the momentum signal with volatility-normalized sizing.

Background: the momentum signal (go LONG the previous 5-day return, hold 1 day,
  daily rebalance) has been admitted as candidate positive evidence in activation
  37318950814, showing a REGIME_STABLE positive edge in 7/10 assets. The lookback-
  sweep check (activation 37361000967) showed the edge is robust across lookback
  3/5/10 in 6/10 assets.

This check evaluates a volatility-normalized sizing rule for the momentum signal.
The position size is scaled by a conservative volatility measure to reduce
leverage-dependent evidence while preserving the directional signal. This tests
whether the momentum signal's robustness survives when position sizing is
normalized by volatility (i.e., when market uncertainty rises, positions are
reduced proportionally).

Falsification prediction: the volatility-normalized momentum will either
  CONSISTENT_WITH_NOISE (no edge anywhere) or REGIME_DEPENDENT (the positive run
  concentrates in one historical regime mix, notably 2009-2013), or
  REGIME_STABLE_LOSS (uniformly negative — the reversal wins on both sides,
  which would be an inconsistency to explain). Either outcome closes the cell.

Method (fixed a-priori, not tuned to OOS):
- Signal: momentum = long the previous 5-day return, hold 1 day, daily rebalance
- Sizing: volatility-normalized = weight * (vol_scale_factor / vol)
  where vol is the rolling realized volatility (20-day lookback)
- vol_scale_factor = 1.0 (conservative, no leverage)
- Conservative parameters prevent any leverage or optimization
- Same walk-forward parameters as the standard momentum check

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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACK = 5
VOL_LOOKBACK = 20
VOL_SCALE_FACTOR = 1.0
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_volatility_normalized_results.json"


def run_asset(asset, tickers):
    """Run momentum check with volatility-normalized sizing on one asset."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    # Run through the regime-stability gate with the volatility-normalized momentum signal
    res = bt.stress_segments(
        signals_fn=bt.momentum_volatility_normalized_signals,
        bars=list(bars),
        signals=bt.momentum_volatility_normalized_signals(closes, LOOKBACK, VOL_LOOKBACK, VOL_SCALE_FACTOR),
        param_grid=[{"lookback": LOOKBACK, "vol_lookback": VOL_LOOKBACK, "vol_scale_factor": VOL_SCALE_FACTOR}],
        baseline=(("lookback", LOOKBACK), ("vol_lookback", VOL_LOOKBACK), ("vol_scale_factor", VOL_SCALE_FACTOR)),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    scen = res.scenarios[0]
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: {'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
    print(f"  dataset: {manifest['dataset_id']} | collection: "
          f"{manifest['collection_date']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
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
        signals = bt.momentum_volatility_normalized_signals(closes, LOOKBACK, VOL_LOOKBACK, VOL_SCALE_FACTOR)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Momentum with volatility-normalized sizing across collected universe ===")
    print("  Long the previous N-day return, hold 1 day, daily rebalanced. "
          "Position size scaled by volatility: weight * (vol_scale_factor / vol), "
          "with vol = rolling realized volatility (20-day lookback). "
          "vol_scale_factor = {} (conservative). Params: walk-forward per segment "
          "train={}d / test={}d / warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments\n".format(VOL_SCALE_FACTOR, TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [ticker][segment index]
    asset_results = {t: run_asset(t, all_tickers) for t in ticker_order}

    per_asset = {}
    verdict_counts = {"CONSISTENT_WITH_NOISE": 0, "REGIME_DEPENDENT": 0,
                      "REGIME_STABLE": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        per_asset[ticker] = asset_results[ticker]
        verdict_counts[asset_results[ticker]["verdict"]] += 1

    print("  Regime gate verdict counts across universe:")
    for verdict, count in verdict_counts.items():
        print(f"    {verdict}: {count}")

    print("\n  Per-asset momentum (volatility-normalized sizing) results:")
    for ticker in ticker_order:
        res = per_asset[ticker]
        medians = "[", ", ".join(f"{m:+.3f}" for m in res["medians"]), "]"
        print(f"    {ticker:6s}: segments={[s for s in res['segments']]}, "
              f"candidate_medians={medians} "
              f"dispersion={res['candidate_dispersion']:+.3f} "
              f"verdict={res['verdict']}")

    # Determinism: rerun the target subset and assert identical output.
    res2 = {t: run_asset(t, tickers) for t in target}
    out1 = json.dumps({t: asset_results[t] for t in target}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        vol_lookback=VOL_LOOKBACK,
        vol_scale_factor=VOL_SCALE_FACTOR,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            verdict_counts=verdict_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    medians=per_asset[t]["medians"],
                    null_medians=per_asset[t]["null_medians"],
                    candidate_dispersion=per_asset[t]["candidate_dispersion"],
                    verdict=per_asset[t]["verdict"],
                    n_folds=per_asset[t]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. Momentum with volatility-normalized "
          "sizing across the collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
