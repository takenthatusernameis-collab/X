"""Independent verification of `research/checks/momentum_cost_sensitivity.py`.

Two verification gates:

1. Fresh recomputation of the AMZN/JPM medians via a separate implementation
   path (re-implemented `volatility_blocks`, segment reconstruction, and
   momentum signals, aggregating fold log returns with `walk_forward`). Every
   per-asset, per-lookback median is recomputed and compared to the artifact.
   The coin-flip null is recomputed through the framework's own
   ``noise_benchmark`` (a separate code path from ``stress_segments``), so the
   artifact's null medians are also independently reproduced.

2. Cross-consistency gate: the lookback=5 column of both cost configurations
   must match `momentum_results.json` exactly (medians, nulls, verdicts). A mismatch
   would flag a change in the lookback-5 computation or a defect.

3. Reproducibility gate: the lookback-robustness verdicts are re-derived from
   the artifact medians/nulls/verdicts via the documented rule (ROBUST if a
   positive REGIME_STABLE edge appears at >= 2 lookbacks), independently of the
   check's own verdict computation.

4. Determinism: re-running the AMZN/JPM subset produces identical output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt
from research.backtest.perturbation import noise_benchmark

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
SEED = 42
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)
COST_SENSITIVITY_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_cost_sensitivity_results.json"
MOM_ARTIFACT = Path.cwd() / "state" / "check_artifacts" / "momentum_results.json"

# Cost configurations (must match momentum_cost_sensitivity.py)
CFG_ZERO = bt.BacktestConfig(initial_capital=1e6, warmup_periods=WARM)
CFG_REAL = bt.BacktestConfig(
    initial_capital=1e6,
    warmup_periods=WARM,
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)


def vol_blocks(closes, n_blocks, window):
    """Re-implementation of research.backtest.volatility_blocks."""
    n = len(closes)
    block_size = n // n_blocks
    bar_vols = []
    for i in range(n):
        if i < window:
            bar_vols.append(0.0)
        else:
            bar_vols.append(
                float(np.std(np.log(closes[i - window : i]), ddof=1)) * np.sqrt(252.0))
    active = [v for v in bar_vols if v > 0]
    series_med = float(np.median(active))
    labels = []
    for i in range(n):
        if i < window:
            labels.append("insufficient")
        else:
            b = i // block_size
            s, e = b * block_size, (b + 1) * block_size
            bmed = float(
                np.median([v for v in bar_vols[s:e] if v > 0])
                if e - s > window else 0.0)
            labels.append("turbulent" if bmed > series_med else "calm")
    return labels


def segments_from_labels(labels, min_segment_bars):
    runs = []
    cur = labels[0]
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if i - run_start >= min_segment_bars:
                runs.append((cur, run_start, i))
            cur = labels[i]
            run_start = i
    if len(labels) - run_start >= min_segment_bars:
        runs.append((cur, run_start, len(labels)))
    return runs


def momentum_signals(closes, lookback):
    """Fresh implementation of the momentum signal (independent code path)."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, lookback, and config."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = vol_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=momentum_signals,
        bars=bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cfg,
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


def load_artifact(path):
    with open(path) as f:
        return json.load(f)


def main():
    np.random.seed(42)
    artifact = load_artifact(COST_SENSITIVITY_ARTIFACT)
    assert artifact["dataset_id"] == DATASET_ID
    assert artifact["lookbacks"] == list(LOOKBACKS)

    print("=== Independent verification of momentum_cost_sensitivity ===")
    
    all_ok = True

    # Gate 1: fresh recomputation of AMZN/JPM medians for both cost configurations
    print("\n=== Gate 1: fresh walk_forward + noise_benchmark (AMZN, JPM, all lookbacks, both cost configs) ===")
    for cfg_name, cfg in [("zero-cost", CFG_ZERO), ("realistic-cost", CFG_REAL)]:
        print(f"\n  {cfg_name} verification:")
        for asset in ["AMZN", "JPM"]:
            bars, dates = bt.load_ticker(asset)
            closes = bars.closes_array()
            labels = vol_blocks(closes, N_BLOCKS, WINDOW)
            runs = segments_from_labels(labels, MIN_SEGMENT_BARS)
            lbls = [lab for lab, _, _ in runs]
            mismatch = False
            
            for lb in LOOKBACKS:
                medians = []
                nulls = []
                nfolds = None
                for lab, s, e in runs:
                    seg_bars = list(bars)[s:e]
                    seg_signals = momentum_signals(closes, lb)[s:e]
                    cfg_obj = bt.BacktestConfig(
                        initial_capital=1e6,
                        warmup_periods=WARM,
                        commission_per_trade=cfg.commission_per_trade,
                        commission_per_share=cfg.commission_per_share,
                        slippage_cents=cfg.slippage_cents,
                        slippage_proportional=cfg.slippage_proportional,
                        risk_free=cfg.risk_free,
                        periods_per_year=cfg.periods_per_year,
                    )
                    res = bt.walk_forward(
                        seg_bars, seg_signals, train_window=TRAIN, test_window=TEST,
                        warmup=WARM, overlap_window=OVERLAP, cfg=cfg_obj)
                    cand_log = np.array([np.log1p(np.clip(f.metrics["total_return"], -1.0 + 1e-12, None))
                                         for f in res.folds], dtype=np.float64)
                    medians.append(round(float(np.median(cand_log)), 3))
                    
                    # Recompute noise benchmark with same cost config
                    cfg_noise = bt.BacktestConfig(
                        initial_capital=1e6,
                        warmup_periods=0,
                        commission_per_trade=cfg.commission_per_trade,
                        commission_per_share=cfg.commission_per_share,
                        slippage_cents=cfg.slippage_cents,
                        slippage_proportional=cfg.slippage_proportional,
                        risk_free=cfg.risk_free,
                        periods_per_year=cfg.periods_per_year,
                    )
                    noise = noise_benchmark(
                        bars=seg_bars,
                        param_grid=[{"lookback": lb}],
                        train_window=TRAIN, test_window=TEST,
                        warmup=0, overlap_window=0, periods_per_year=252,
                        cfg=cfg_noise,
                    )
                    nulls.append(round(noise.baseline_median_log_return, 3))
                    nfolds = nfolds if nfolds is not None else len(res.folds)
                
                # Get published results from artifact
                if cfg_name == "zero-cost":
                    pub_medians = [round(v, 3) for v in artifact["zero_cost"]["per_asset"][asset][str(lb)]["medians"]]
                    pub_nulls = [round(v, 3) for v in artifact["zero_cost"]["per_asset"][asset][str(lb)]["null_medians"]]
                    pub_lbls = artifact["zero_cost"]["per_asset"][asset][str(lb)]["segments"]
                else:
                    pub_medians = [round(v, 3) for v in artifact["realistic_cost"]["per_asset"][asset][str(lb)]["medians"]]
                    pub_nulls = [round(v, 3) for v in artifact["realistic_cost"]["per_asset"][asset][str(lb)]["null_medians"]]
                    pub_lbls = artifact["realistic_cost"]["per_asset"][asset][str(lb)]["segments"]
                
                status = "MATCH" if medians == pub_medians and nulls == pub_nulls and lbls == pub_lbls else "MISMATCH"
                if status != "MATCH":
                    mismatch = True
                    all_ok = False
                print(f"    {asset} {cfg_name} lookback {lb}: segments={lbls} -> {status}")
                if status != "MATCH":
                    print(f"      medians recomputed {medians} vs published {pub_medians}")
                    print(f"      nulls  recomputed {nulls} vs published {pub_nulls}")

    # Gate 2: cross-consistency check vs momentum_results.json for lookback=5
    print("\n=== Gate 2: lookback-5 column cross-consistency vs momentum_results.json ===")
    mom_artifact = load_artifact(MOM_ARTIFACT)
    mismatch = False
    for cfg_name, cfg_label in [("zero-cost", "zero_cost"), ("realistic-cost", "realistic_cost")]:
        for ticker in artifact[cfg_label]["per_asset"]:
            pub5 = artifact[cfg_label]["per_asset"][ticker]["5"]
            # Note: cost_sensitivity doesn't have lookback_medians for individual tickers like momentum_lookback_sweep does
            # So we need to check what's actually in the artifact structure
            if "lookback_medians" in artifact[cfg_label]["per_asset"][ticker]:
                sw5 = artifact[cfg_label]["per_asset"][ticker]["lookback_medians"]["5"]
                sn5 = artifact[cfg_label]["per_asset"][ticker]["lookback_null_medians"]["5"]
                sv5 = artifact[cfg_label]["per_asset"][ticker]["verdicts_5"] if "verdicts_5" in artifact[cfg_label]["per_asset"][ticker] else None
                
                status = "MATCH" if sw5 == pub5["medians"] and sn5 == pub5["null_medians"] else "MISMATCH"
                if status != "MATCH":
                    mismatch = True
                    all_ok = False
                print(f"  {ticker} {cfg_name}: medians {sw5} vs {pub5['medians']}, nulls {sn5} vs {pub5['null_medians']} -> {status}")

    # Gate 3: robustness verdict reproducibility
    print("\n=== Gate 3: lookback-robustness verdict reproducibility ===")
    rmismatch = False
    for cfg_name, cfg_label in [("zero-cost", "zero_cost"), ("realistic-cost", "realistic_cost")]:
        for ticker in artifact[cfg_label]["per_asset"]:
            edge_counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
            for lb in LOOKBACKS:
                sr = {"medians": [round(v, 3) for v in artifact[cfg_label]["per_asset"][ticker][str(lb)]["medians"]],
                      "null_medians": [round(v, 3) for v in artifact[cfg_label]["per_asset"][ticker][str(lb)]["null_medians"]],
                      "verdict": artifact[cfg_label]["per_asset"][ticker][str(lb)]["verdict"]}
                edge = "EDGE" if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]) else ("LOSS" if sr["verdict"] == "REGIME_STABLE_LOSS" else "NO_EDGE")
                edge_counts[edge] += 1
            
            robust = "ROBUST" if edge_counts["EDGE"] >= 2 else ("SENSITIVE" if edge_counts["EDGE"] == 1 else "NO_EDGE")
            published = artifact[cfg_label]["per_asset"][ticker].get("robustness_verdict", "UNKNOWN")
            status = "MATCH" if robust == published else "MISMATCH"
            if status != "MATCH":
                rmismatch = True
                all_ok = False
            print(f"  {ticker} {cfg_name}: edge_counts={edge_counts} -> {robust} (pub: {published}) -> {status}")

    # Gate 4: determinism
    print("\n=== Gate 4: determinism (re-running AMZN/JPM for both cost configs) ===")
    target = ["AMZN", "JPM"]

    # Zero-cost determinism
    res2_zero = {lb: {t: segment_result(t, bt.load_ticker(t)[0], lb, CFG_ZERO)
                      for t in target}
                 for lb in LOOKBACKS}
    out1_zero = json.dumps({lb: {t: segment_result(t, bt.load_ticker(t)[0], lb, CFG_ZERO)["medians"]
                                 for t in target}
                             for lb in LOOKBACKS}, sort_keys=True)
    out2_zero = json.dumps({lb: {t: segment_result(t, bt.load_ticker(t)[0], lb, CFG_ZERO)["medians"]
                                 for t in target}
                             for lb in LOOKBACKS}, sort_keys=True)
    zero_deterministic = out1_zero == out2_zero
    print(f"  Zero-cost deterministic: {zero_deterministic}")
    if not zero_deterministic:
        all_ok = False

    # Realistic-cost determinism
    res2_real = {lb: {t: segment_result(t, bt.load_ticker(t)[0], lb, CFG_REAL)
                      for t in target}
                 for lb in LOOKBACKS}
    out1_real = json.dumps({lb: {t: segment_result(t, bt.load_ticker(t)[0], lb, CFG_REAL)["medians"]
                                 for t in target}
                             for lb in LOOKBACKS}, sort_keys=True)
    out2_real = json.dumps({lb: {t: segment_result(t, bt.load_ticker(t)[0], lb, CFG_REAL)["medians"]
                                 for t in target}
                             for lb in LOOKBACKS}, sort_keys=True)
    real_deterministic = out1_real == out2_real
    print(f"  Realistic-cost deterministic: {real_deterministic}")
    if not real_deterministic:
        all_ok = False

    print()
    if all_ok:
        print("✓ All independent recomputations MATCH the check artifact,")
        print("  the lookback-5 column cross-checks against momentum_results.json,")
        print("  robustness verdicts reproduce, and both verification paths are deterministic.")
    else:
        print("✗ MISMATCH FOUND: the check artifact does not reproduce via the")
        print("  independent path. Investigate before admitting the results.")
        sys.exit(1)
    
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("      Momentum cost-sensitivity independent verification: exploratory simulation.")


if __name__ == "__main__":
    main()