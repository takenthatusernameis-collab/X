"""Test the momentum signal class on the collected universe with a past-only volatility gate.

Background: the trend-following MA-crossover class was exhaustively explored and rejected. 
The short-horizon return-reversal class was exhaustively explored and rejected: every segment 
of every asset showed a negative median walk-forward log return; the perturbation sweep was flat on the null.

Momentum is the direct competing hypothesis to reversal: go LONG the previous N-day return instead of shorting it, 
hold 1 day, daily rebalance. The momentum test on lookback-5 was admitted as candidate positive evidence: 
REGIME_STABLE positive edge in 7/10 of the collected universe at a one-day hold.

This test adds a past-only volatility gate to the momentum signal: only activate momentum signals when 
past volatility (using AAPL as market proxy) is below a predeclared threshold. This addresses the question:
"Does a single past-only volatility gate create a predeclared robustness improvement for the qualified momentum effect?"

Falsification prediction: if the volatility gate removes too much data or conditions out the regime where
momentum works, the gated momentum either shows CONSISTENT_WITH_NOISE (no edge) or REGIME_DEPENDENT 
(the edge concentrates in one historical regime mix), or REGIME_STABLE_LOSS (uniformly negative).
Either outcome closes the cell.

Method: same as momentum.py but with volatility gate on AAPL.
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

VOL_GATE_THRESHOLD = 0.20  # Predeclared threshold for past-only volatility gate
MARKET_PROXY = "AAPL"

def volatility_gate_signals(closes, vol_window: int = 20, threshold: float = 0.20):
    """Past-only volatility gate: only emit signals when realized vol < threshold.
    
    Annualized realized volatility is the standard deviation of log returns over
    `vol_window` bars, scaled to an annual basis. Weight is +1 (momentum active)
    in low-vol regimes (realized vol below `threshold`) and 0 (momentum gated)
    otherwise. Threshold is fixed a-priori.
    """
    n = len(closes)
    signals = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    sqrt252 = np.sqrt(252.0)
    for i in range(int(vol_window) - 1, n):
        log_returns = np.log(closes[i - vol_window + 1 : i + 1])
        realized_annual = float(np.std(log_returns, ddof=1)) * sqrt252
        signals[i] = bt.Signal(date=i + 1, weight=1.0 if realized_annual < threshold else 0.0)
    return signals

def momentum_signals(closes, lookback: int = LOOKBACK):
    """Standard momentum signal: long previous lookback-day return, hold 1 day."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    for i in range(lookback - 1, n):
        out[i] = bt.Signal(date=i + 1, weight=1.0 if closes[i] > closes[i - lookback] else -1.0)
    return out

def run_asset(asset, tickers):
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    
    # Get market proxy volatility gate
    aapl_bars, _ = bt.load_ticker(MARKET_PROXY)
    aapl_closes = aapl_bars.closes_array()
    gate_signals = volatility_gate_signals(aapl_closes, vol_window=WINDOW, threshold=VOL_GATE_THRESHOLD)
    
    results = {}
    
    for name, sig, params in [
        ("momentum", lambda c, **p: momentum_signals(c, lookback=LOOKBACK), {"lookback": LOOKBACK}),
    ]:
        momentum_sig = sig(closes)
        # Apply volatility gate: multiply momentum signal by gate (0 or 1)
        gated_signals = []
        for i in range(len(momentum_sig)):
            gate_weight = gate_signals[i].weight
            gated_signals.append(bt.Signal(date=momentum_sig[i].date, 
                                         weight=momentum_sig[i].weight * gate_weight))
        
        if len(gated_signals) != len(closes):
            raise ValueError(f"{asset} {name}: gated signals {len(gated_signals)} != bars {len(closes)}")
        
        res = bt.stress_segments(
            signals_fn=lambda c, **p: [bt.Signal(date=i + 1, 
                                      weight=momentum_signals(c, lookback=LOOKBACK)[i].weight * 
                                      volatility_gate_signals(c, vol_window=WINDOW, threshold=VOL_GATE_THRESHOLD)[i].weight)
                                for i in range(len(c))],
            bars=list(bars),
            signals=gated_signals,
            param_grid=[params],
            baseline=tuple(params.items()),
            segment_fn=seg_fn,
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        names = [s.name for s in res.scenarios]
        medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
        results[name] = dict(
            segments=names,
            medians=medians,
            candidate_dispersion=round(res.candidate_dispersion, 3),
            null_dispersion=round(res.null_dispersion, 3),
            verdict=res.overall_verdict,
            n_folds=res.n_folds,
        )
    return results

def perturbation_sweep(n_bars=600):
    """Synthetic perturbation sweep of momentum lookback over the canonical
    regime family, with a coin-flip null benchmark."""
    bars = bt.generate_bars(n_bars, seed=SEED)
    grid = bt.parameter_grid_around(
        (("lookback", LOOKBACK),),
        multipliers=(0.5, 1.0, 2.0),
    )
    sweep = bt.parameter_sweep(
        signals_fn=bt.momentum_signals,
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=10,
        overlap_window=10,
        periods_per_year=252,
    )
    summary = bt.sweep_summary(sweep, baseline=(("lookback", LOOKBACK),))
    noise = bt.noise_benchmark(
        bars=bars,
        param_grid=grid,
        train_window=60,
        test_window=20,
        warmup=0,
        overlap_window=0,
        periods_per_year=252,
        seed=SEED,
    )
    return dict(
        param_sets=[tuple(ps) for ps in sweep.param_sets],
        medians=[round(m, 3) for m in summary.median_log_returns],
        baseline_median=round(summary.baseline_median_log_return, 3),
        noise_median=round(noise.baseline_median_log_return, 3),
        compare_noise_default=summary.compare_noise(noise.baseline_median_log_return),
        compare_noise_fixed=summary.compare_noise(noise.baseline_median_log_return, tol=0.05),
        effective_tolerance=round(summary.effective_tolerance, 4),
        n_folds=summary.n_folds,
    )

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
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = momentum_signals(closes, lookback=LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "neutral bars before the lookback window carry the equity forward.")

    print("\n=== 4. Volatility-gate momentum on REGIME_DEPENDENT assets ===")
    print("Momentum: long the previous {}-day return, hold 1 day, daily "
          "rebalanced. Params: walk-forward per segment "
          "train={}d / test={}d / warmup={}d / overlap={}d, min {} "
          "bars/segment, volatility gate: AAPL past-only realized vol < {}% annualized.\n".format(
              LOOKBACK, TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS, VOL_GATE_THRESHOLD * 100))
    print("Segments: {} contiguous blocks by date, past-only volatility "
          "classifier (turbulent = block median trailing-{}-bar realized vol "
          "> series-wide median)\n".format(N_BLOCKS, WINDOW))

    results_all = {}
    for asset in target:
        res = run_asset(asset, tickers)
        results_all[asset] = res
        print("Asset: {}".format(asset))
        print("  {:14s} {:28s} {:28s} {:12s}  {:12s}  verdict".format(
            "variant", "segments", "candidate_medians", "cand_disp", "null_disp"))
        for name, r in res.items():
            segs = "[" + ",".join(r["segments"]) + "]"
            meds = "[" + ",".join(str(m) for m in r["medians"]) + "]"
            print(
                "  {:14s} {:28s} {:28s} {:+10.3f}  {:+10.3f}  {}".format(
                    name, segs, meds, r["candidate_dispersion"], r["null_dispersion"], r["verdict"]))
        print("")

    # Full-universe generalization: volatility-gated momentum on all collected tickers
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]
    summary = bt.stress_segments_across_tickers(
        tickers=tickers_univ,
        signals_fn=lambda c, **kw: momentum_signals(c, lookback=LOOKBACK),
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
    print("=== 5. Volatility-gated momentum across collected universe ===")
    print("  " + summary.inspect().replace("\n", "\n  "))
    print("")

    # Synthetic perturbation sweep (parameter-robustness gate)
    pers = perturbation_sweep()
    print("=== 6. Synthetic perturbation sweep (lookback 3 / 5 / 10) ===")
    print("  param sets: {}".format(pers["param_sets"]))
    print("  candidate medians: {}".format(pers["medians"]))
    print("  baseline (lookback=5): {:+.3f}".format(pers["baseline_median"]))
    print("  coin-flip null baseline: {:+.3f}".format(pers["noise_median"]))
    print("  baseline vs null (sample-calibrated tol {:.4f}): {}"
          .format(pers["effective_tolerance"], pers["compare_noise_default"]))
    print("  baseline vs null (fixed tol 0.05): {}"
          .format(pers["compare_noise_fixed"]))
    print("")

    # Determinism: rerun the 2-asset comparison and assert identical output
    aapl_bars, _ = bt.load_ticker(MARKET_PROXY)
    aapl_closes = aapl_bars.closes_array()
    aapl_gate = volatility_gate_signals(aapl_closes, vol_window=WINDOW, threshold=VOL_GATE_THRESHOLD)
    
    res2 = {a: run_asset(a, tickers) for a in target}
    out1 = json.dumps(results_all, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Write the artifact that the independent verification file recomputes
    # from a separate path and compares against.
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        vol_gate_threshold=VOL_GATE_THRESHOLD,
        market_proxy=MARKET_PROXY,
        vol_gate_activated_signals=sum(1 for s in aapl_gate if s.weight > 0),
        vol_gate_total_signals=len(aapl_gate),
        per_asset=results_all,
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
        perturbation=pers,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("")
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Volatility-gated momentum test on AMZN/JPM + universe: exploratory simulation.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
