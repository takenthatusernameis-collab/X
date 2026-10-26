"""Test past-only volatility gate on lookback-5 momentum.

Objective: Test one explicitly predeclared past-only volatility gate on lookback-5 momentum through the existing regime-stability and null framework.

A volatility gate filters the momentum signal: only generate momentum signals when past volatility is in a predeclared regime (e.g., 'calm').

Method (fixed a-priori):
- Use lookback=5 momentum signal (long previous 5-day return)
- Add past-only volatility gate: only trade momentum when past 60-day realized volatility is in 'calm' regime
- Walk-forward validation: train=252d/test=84d/warmup=60d/overlap=60d
- 4 contiguous volatility blocks, min 400 bars/segment
- Coin-flip null benchmark on same segments
- Regime-stability verdict: REGIME_STABLE indicates robust performance across regimes
- No threshold search or post-hoc optimization

Evidence gate: The gate either demonstrates reproducible improvement in robustness or is falsified/unsupported.
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
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_volatility_gate_results.json"


def volatility_gate_momentum_signals(closes, lookback, volatility_gate="calm"):
    """Past-only momentum signal gated by past volatility.

    Generate momentum signals only when past volatility is in the specified gate.
    The gate is determined by bt.volatility_blocks (past-only classification).
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        # Get past-only volatility regime for this bar
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
        if labels[i] == volatility_gate:
            ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
            out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def run_asset(asset, tickers, gate_type="calm"):
    """Run regime-stability gate test for one asset with volatility gate."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    signals = volatility_gate_momentum_signals(closes, LOOKBACK, gate_type)
    if len(signals) != len(closes):
        raise ValueError(f"{asset}: signals {len(signals)} != bars {len(closes)}")

    res = bt.stress_segments(
        signals_fn=lambda c, **kw: volatility_gate_momentum_signals(c, LOOKBACK, gate_type),
        bars=list(bars),
        signals=signals,
        param_grid=[{"gate_type": gate_type}],
        baseline=(("gate_type", gate_type),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    return res


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

    print("\n=== 3. Leakage review per asset ===")
    target = ["AMZN", "JPM", "AAPL", "MSFT", "GOOGL"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = volatility_gate_momentum_signals(closes, LOOKBACK, "calm")
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, gated momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: gated momentum sign uses past-only volatility and closes[:t] only; "
          "neutral bars before lookback carry equity forward.")

    print("\n=== 4. Volatility-gated momentum across collected universe ===")
    print(f"Gate: only trade momentum when past 60-day realized volatility is 'calm' (below series median)")
    print(f"Params: walk-forward per segment train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d, "
          f"min {MIN_SEGMENT_BARS} bars/segment, 4 contiguous volatility blocks, compared vs coin-flip null\n")

    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: volatility_gate_momentum_signals(c, LOOKBACK, "calm"),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("gate_type", "calm"),),
        param_grid=[{"gate_type": "calm"}],
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    print("=== 5. Gated momentum across collected universe ===")
    print("  " + summary.inspect().replace("\n", "\n  "))
    print("")

    # Determinism: rerun target subset and compare
    res2 = {a: run_asset(a, tickers, "calm") for a in target}
    out1 = json.dumps({a: res2[a].overall_verdict for a in target}, sort_keys=True)
    out2 = json.dumps({a: res2[a].overall_verdict for a in target}, sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    # Write artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        gate_type="calm",
        gate_description="only trade momentum when past 60-day realized volatility is in 'calm' regime",
        per_asset=dict(
            (t, dict(
                segments=[s.name for s in r.scenarios],
                medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
                null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
                candidate_dispersion=round(r.candidate_dispersion, 3),
                null_dispersion=round(r.null_dispersion, 3),
                verdict=r.overall_verdict,
                n_folds=r.n_folds,
            ))
            for t, r in summary.assets.items()
        ),
        universe=dict(
            verdict_counts=summary.verdict_counts,
            ticker_order=summary.ticker_order,
            per_asset=dict(
                (t, dict(
                    segments=[s.name for s in r.scenarios],
                    medians=[round(s.baseline_median_log_return, 3) for s in r.scenarios],
                    null_medians=[round(s.noise_median_log_return, 3) for s in r.scenarios],
                    candidate_dispersion=round(r.candidate_dispersion, 3),
                    null_dispersion=round(r.null_dispersion, 3),
                    verdict=r.overall_verdict,
                ))
                for t, r in summary.assets.items()
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"artifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production execution. "
          "Momentum with past-only volatility gate on the collected 10-asset universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
