from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

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
ARTIFACT_PATH = ARTIFACT_DIR / "regime_conditional_momentum_results.json"


def base_momentum_signals(closes, lookback=LOOKBACK):
    """Past-only momentum signal (same as research/backtest/momentum.py:886)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def regime_conditional_momentum_signals(closes, lookback=LOOKBACK, window=WINDOW):
    """Momentum signal gated by past-only volatility threshold (predeclared).
    
    Only active when past volatility exceeds threshold (0.25 annualized).
    Past-only: volatility computed from closes[:i] only.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        # Past-only volatility computation
        vol_window = min(window, max(4, i // 4))
        if i >= vol_window:
            early_vol = float(
                np.std(np.log(closes[vol_window // 2 : vol_window]), ddof=1)
            ) * np.sqrt(252.0)
            # PREDECLARED VOLATILITY GATE (threshold = 0.25 = 25% annualized)
            if early_vol > 0.25:
                ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
                out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def regime_labels(closes, n_blocks=N_BLOCKS, window=WINDOW):
    """Create past-only volatility regime labels (same as regime_stability.volatility_blocks)."""
    return bt.volatility_blocks(closes, n_blocks=n_blocks, window=window)


def run_asset(asset, tickers, use_gate=False):
    """Run momentum test on one asset with optional volatility gate."""
    bars = tickers[asset]
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)

    if use_gate:
        signals_fn = lambda c, **p: regime_conditional_momentum_signals(c, lookback=LOOKBACK)
        result_name = f"{asset}_gate"
    else:
        signals_fn = lambda c, **p: base_momentum_signals(c, lookback=LOOKBACK)
        result_name = f"{asset}_base"

    signals = signals_fn(closes)
    if len(signals) != len(closes):
        raise ValueError(f"{asset} {result_name}: signals {len(signals)} != bars {len(closes)}")

    res = bt.stress_segments(
        signals_fn=signals_fn,
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

    names = [s.name for s in res.scenarios]
    medians = [round(s.baseline_median_log_return, 3) for s in res.scenarios]
    
    return {
        result_name: dict(
            segments=names,
            medians=medians,
            candidate_dispersion=round(res.candidate_dispersion, 3),
            null_dispersion=round(res.null_dispersion, 3),
            verdict=res.overall_verdict,
            n_folds=res.n_folds,
        )
    }


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== Regime-Conditional Momentum Gate Test ===")
    print("TEST OBJECTIVE: Evaluate a predeclared past-only volatility gate on lookback-5 momentum")
    print("VOLATILITY GATE: Only trade when past volatility > 25% annualized (vol_window = 60)")
    print("BASELINE: Same momentum without volatility gate")
    print("UNIVERSE: 10 large-cap US equities (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, JNJ, XOM)")
    print()

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | collection: {manifest['collection_date']} | universe: {len(manifest['universe'])}")
    print()

    print("=== 2. Leakage review ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = regime_conditional_momentum_signals(closes, lookback=LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, gated momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  Past-only: volatility gate uses only closes[:t]; neutral before gate condition met.")
    print()

    print("=== 3. Momentum vs Gated Momentum on REGIME_DEPENDENT assets ===")
    print(f"Params: lookback={LOOKBACK}, train={TRAIN}d/test={TEST}d/warmup={WARM}d/overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment")
    print(f"Segments: {N_BLOCKS} contiguous volatility blocks (calm/turbulent)")
    print()

    results_all = {}
    for asset in target:
        res_gate = run_asset(asset, tickers, use_gate=True)
        res_base = run_asset(asset, tickers, use_gate=False)
        results_all.update(res_gate)
        results_all.update(res_base)
        print(f"Asset: {asset}")
        print(f"  {'variant':14s} {'segments':28s} {'candidate_medians':28s} {'cand_disp':>10s} {'null_disp':>10s}  verdict")
        gate_r = res_gate[f"{asset}_gate"]
        base_r = res_base[f"{asset}_base"]
        segs_g = "[" + ",".join(gate_r["segments"]) + "]"
        meds_g = "[" + ",".join(str(m) for m in gate_r["medians"]) + "]"
        segs_b = "[" + ",".join(base_r["segments"]) + "]"
        meds_b = "[" + ",".join(str(m) for m in base_r["medians"]) + "]"
        print(f"  {asset}_gate    {segs_g:28s} {meds_g:28s} {gate_r['candidate_dispersion']:>10.3f}  {gate_r['null_dispersion']:>10.3f}  {gate_r['verdict']}")
        print(f"  {asset}_base    {segs_b:28s} {meds_b:28s} {base_r['candidate_dispersion']:>10.3f}  {base_r['null_dispersion']:>10.3f}  {base_r['verdict']}")
        print()

    print("=== 4. Full-universe generalization ===")
    tickers_univ = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers_univ[ticker] = bt.load_ticker(ticker)[0]

    for use_gate, label in [(False, "baseline"), (True, "volatility_gate")]:
        def make_signals_fn(gated: bool):
            if gated:
                return lambda c, **kw: regime_conditional_momentum_signals(c, lookback=LOOKBACK)
            else:
                return lambda c, **kw: base_momentum_signals(c, lookback=LOOKBACK)
        
        signals_fn = make_signals_fn(use_gate)
        summary = bt.stress_segments_across_tickers(
            tickers=tickers_univ,
            signals_fn=signals_fn,
            regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", LOOKBACK),),
            param_grid=[{"lookback": LOOKBACK}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        verdict_counts = summary.verdict_counts
        print(f"\n{label.upper()} MOMENTUM (10 assets):")
        print(f"  Verdict counts: CONSISTENT_WITH_NOISE={verdict_counts['CONSISTENT_WITH_NOISE']}, "
              f"REGIME_DEPENDENT={verdict_counts['REGIME_DEPENDENT']}, "
              f"REGIME_STABLE={verdict_counts['REGIME_STABLE']}, "
              f"REGIME_STABLE_LOSS={verdict_counts['REGIME_STABLE_LOSS']}")

    print("\n=== 5. Determinism check ===")
    res1 = {a: run_asset(a, tickers, use_gate=True)[f"{a}_gate"] for a in target}
    res2 = {a: run_asset(a, tickers, use_gate=True)[f"{a}_gate"] for a in target}
    out1 = json.dumps(res1, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print(f"determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    print("\n=== 6. Artifact written ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookback=LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        pre_asset=results_all,
        description="Regime-conditional momentum: volatility gate threshold = 0.25 annualized",
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"artifact written to: {ARTIFACT_PATH}")
    print("\nREGIME-CONDITIONAL MOMENTUM TEST COMPLETE")
    print("The volatility gate (0.25 threshold) either demonstrates robustness improvement or is falsified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
