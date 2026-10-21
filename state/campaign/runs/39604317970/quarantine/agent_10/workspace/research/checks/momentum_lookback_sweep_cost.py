"""Real-data cost-sensitive momentum lookback sweep across the collected universe.

Objective: test whether the momentum edge observed at lookback=5
(`research/checks/momentum.py`, activation 37318950814) survives realistic
transaction costs and persists across lookback horizons.

For each collected ticker, momentum (long the previous N-day return, hold 1 day,
daily rebalance) is walk-forward validated with costs (commission + slippage)
under the same walk-forward windows and volatility blocks, for lookback in
{3, 5, 10}. Each lookback's per-segment median log return is compared against
a coin-flip null benchmark run on the same segments but with identical costs.
The edge must survive both the cost layer and the horizon robustness test
before the momentum class can be admitted to the evidence base as candidate
positive evidence.

Method (fixed a-priori, not tuned to OOS):
- Realistic transaction costs applied to every round trip (same as real-data
  execution, not zero). Uses the same walk-forward windows and volatility
  blocks as `momentum_lookback_sweep.py`. Determinism is asserted via an
  internal independent recomputation of the AMZN/JPM subset.

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
LOOKBACKS = (3, 5, 10)
# Realistic transaction cost parameters
# Based on industry-standard retail trading costs
COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    # Commission: $0.01 per trade + $0.005 per share (round-trip)
    commission_per_trade=0.01,
    commission_per_share=0.005,
    # Slippage: 5 cents fixed + 5 bps proportional
    slippage_cents=5.0,
    slippage_proportional=0.0005,
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_cost_results.json"


def segment_result_with_costs(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback, with costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=COST_CONFIG,
    )
    scen = res.scenarios[0]
    # Calculate actual cost impact
    total_commission = sum(f.commission for f in res.trades)
    total_slippage = sum(f.shares * f.price * COST_CONFIG.slippage_proportional for f in res.trades if f.shares != 0)
    cost_impact = (total_commission + total_slippage) / 1e6  # Normalize by initial capital
    
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
        cost_impact=round(cost_impact, 6),
    )


def volatility_gate_momentum_signals(closes, lookback, labels=None):
    """Past-only momentum signal with volatility regime gate.
    
    Activates momentum signals only in turbulent volatility regimes.
    This is the predeclared volatility gate being tested.
    
    Args:
        closes: closing prices array
        lookback: lookback window for momentum calculation
        labels: precomputed volatility labels (if None, computed fresh)
        
    Returns:
        list of Signal objects with momentum only in turbulent regimes
    """
    if labels is None:
        # Use past-only volatility blocks classification
        labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    
    for i in range(lookback, n):
        # Only compute momentum in turbulent regimes
        if labels[i] == "turbulent":
            ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
            out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result_with_volatility_gate(ticker, bars, lookback):
    """Walk-forward regime-stability result for momentum with volatility gate."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = volatility_gate_momentum_signals(closes, lookback=lookback, labels=labels)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=lambda c, **p: volatility_gate_momentum_signals(c, lookback=lookback, labels=labels),
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=COST_CONFIG,
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


def lookback_verdict(sr):
    """Robustness verdict for one lookback's scenario result.

    REGIME_STABLE with uniformly positive medians means this lookback shows a
    stable positive edge on this asset; REGIME_STABLE with uniformly negative
    medians is a stable loss, not an edge.
    """
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def robustness_verdict(lookback_results):
    """Verdict on whether the edge is lookback-robust for one asset.

    ROBUST      : uniformly positive REGIME_STABLE edges at >= 2 lookbacks.
    SENSITIVE   : a positive REGIME_STABLE edge at exactly one lookback only.
    NO_EDGE     : no lookback shows a positive REGIME_STABLE edge.
    """
    counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
    for lb in LOOKBACKS:
        counts[lookback_verdict(lookback_results[str(lb)])] += 1
    if counts["EDGE"] >= 2:
        return "ROBUST", counts
    if counts["EDGE"] == 1:
        return "SENSITIVE", counts
    return "NO_EDGE", counts


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

    print("\n=== 3. Cost model applied ===")
    print("  Commission: $0.01 per trade + $0.005 per share")
    print("  Fixed slippage: 5 cents per share")
    print("  Proportional slippage: 5 bps")
    print("  Total cost impact will be measured per asset/segment")

    print("\n=== 4. Momentum lookback sweep (3 / 5 / 10) across collected universe ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, each lookback vs a coin-flip null "
          "on the same segments with identical costs\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep with costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index].
    lookback_results = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb) for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        profile = {}
        for lb in LOOKBACKS:
            lb_str = str(lb)
            profile[lb_str] = lookback_results[lb][ticker]
            # Store lookback_medians and lookback_null_medians for artifact
            profile[f"lookback_{lb}_medians"] = profile[lb_str]["medians"]
            profile[f"lookback_{lb}_null_medians"] = profile[lb_str]["null_medians"]
        
        v, counts = robustness_verdict({str(lb): profile[str(lb)] for lb in LOOKBACKS})
        robustness_counts[v] += 1
        r5 = profile["5"]
        lookback_5_counts[r5["verdict"]] += 1
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        per_asset[ticker] = profile

    print("  lookback 5 (reference) verdict counts across universe: "
          f"{lookback_5_counts}")
    print("  lookback-robustness verdict counts: "
          f"ROBUST={robustness_counts['ROBUST']} "
          f"(>=2 lookbacks positive edge) | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")
    print("  Per-asset perturbation profiles (costs applied):")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
    print("    {:6s} segments={:<8} medians3510={:<40s} rob={:<8s} "
               "edge_counts={}".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["edge_counts"]))
    print("")

    # Prepare deterministic reference: target subset for AMZN and JPM
    tickers = {}
    for ticker in ["AMZN", "JPM"]:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars

    print("\n=== 5. Volatility gate test on lookback-5 momentum ===")
    print("  Past-only volatility gate: only active in turbulent regimes (vol > median).")
    print("  Compare against lookback-5 momentum without gate.")
    print("  If gate improves robustness or creates edge, momentum benefits from regime conditioning.")
    print("  If gate falsifies (no edge or worse than baseline), past-only volatility classifier")
    print("  cannot separate profitable from non-profitable turbulent regimes for momentum.")
    print("  Results: lookback=5, train={}d / test={}d / warmup={}d / overlap={}d\n".format(TRAIN, TEST, WARM, OVERLAP))

    vol_gate_results = {}
    for ticker in ["AMZN", "JPM"]:
        vol_gate_results[ticker] = {}
        for lb in LOOKBACKS:
            if lb == 5:  # Focus on lookback-5 for the volatility gate test
                res = segment_result_with_volatility_gate(ticker, tickers[ticker], lb)
                vol_gate_results[ticker][str(lb)] = res
                print("  {}: lookback={}, segments={}, medians={}, verdict={}, cost_impact={:+.6f}".format(
                    ticker, lb, res["segments"], res["medians"], res["verdict"], res.get("cost_impact", 0.0)))

    # Compare with baseline momentum
    print("\n=== 6. Volatility gate vs baseline momentum ===")
    print("  Comparison for lookback-5 on AMZN and JPM:")
    for ticker in ["AMZN", "JPM"]:
        baseline = lookback_results[5][ticker]
        gate_result = vol_gate_results[ticker]["5"]
        
        baseline_verdict = baseline["verdict"]
        gate_verdict = gate_result["verdict"]
        
        baseline_medians = baseline["medians"]
        gate_medians = gate_result["medians"]
        
        baseline_cost = baseline.get("cost_impact", 0.0)
        gate_cost = gate_result.get("cost_impact", 0.0)
        
        verdict_changed = "NO" if baseline_verdict == gate_verdict else "YES"
        medians_different = baseline_medians != gate_medians
        
        print("  {}:".format(ticker))
        print("    Baseline momentum: {}".format(baseline_verdict))
        print("    With volatility gate: {}".format(gate_verdict))
        print("    Verdict change: {}; Medians changed: {}".format(verdict_changed, medians_different))
        
        if baseline_verdict == "REGIME_DEPENDENT" and gate_verdict in ["REGIME_STABLE", "REGIME_STABLE_LOSS"]:
            print("    -> VOLATILITY GATE IMPROVES ROBUSTNESS")
        elif baseline_verdict == "REGIME_STABLE" and gate_verdict in ["REGIME_DEPENDENT", "CONSISTENT_WITH_NOISE"]:
            print("    -> VOLATILITY GATE HARMS ROBUSTNESS")
        elif baseline_verdict != gate_verdict:
            print("    -> VOLATILITY GATE CHANGES VERDICT")
        else:
            print("    -> VOLATILITY GATE NO EFFECT")

    # Prepare artifact with volatility gate results
    artifact_vol_gate = {}
    for ticker in ["AMZN", "JPM"]:
        artifact_vol_gate[ticker] = {
            "lookback_5_medians": vol_gate_results[ticker]["5"]["medians"],
            "lookback_5_null_medians": vol_gate_results[ticker]["5"]["null_medians"],
            "lookback_5_verdict": vol_gate_results[ticker]["5"]["verdict"],
        }

    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: segment_result_with_costs(t, all_tickers[t], lb) for t in ["AMZN", "JPM"]}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in ["AMZN", "JPM"]}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Combine results into a single artifact
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_config={
            "commission_per_trade": COST_CONFIG.commission_per_trade,
            "commission_per_share": COST_CONFIG.commission_per_share,
            "slippage_cents": COST_CONFIG.slippage_cents,
            "slippage_proportional": COST_CONFIG.slippage_proportional,
        },
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t][f"lookback_{lb}_medians"]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t][f"lookback_{lb}_null_medians"]
                                         for lb in LOOKBACKS},
                    verdicts_5=per_asset[t]["5"]["verdict"],
                    edge_counts=per_asset[t]["edge_counts"],
                    robustness_verdict=per_asset[t]["robustness_verdict"],
                    n_folds_5=per_asset[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
        # Volatility gate results for lookback-5 momentum
        volatility_gate=dict(
            amzn=artifact_vol_gate["AMZN"],
            jpm=artifact_vol_gate["JPM"],
            test_summary="REGIME_DEPENDENT baseline with past-only volatility gate (turbulent only). "
                        "If gate improves robustness or creates edge, momentum benefits from regime conditioning. "
                        "If gate falsifies, past-only volatility classifier cannot separate profitable from non-profitable turbulent regimes."
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback sweep with realistic costs across the "
          "collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())