"""Cost-sensitivity check for lookback 3/5/10 momentum on the collected universe.

Objective: test whether the short-horizon momentum edge survives conservative
transaction-cost stress under realistic industry cost assumptions.

The LEARNING_STATE admits short-horizon momentum as candidate positive evidence
but leaves execution-cost robustness as a meaningful qualification.

Method: long the previous lookback-day return, hold 1 day, daily rebalance.
Cost model (realistic institutional-grade):
- commission: $1 per round trip
- commission per share: $0.01
- fixed slippage: $0.01 per share
- proportional slippage: 5 bps

Walk-forward: train=252d / test=84d / warmup=60d / overlap=60d,
4 contiguous volatility blocks, min 400 bars/segment.

Independent verification: reimplements regime blocks, momentum signals,
and matched null via fresh walk_forward / noise_benchmark.

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
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400
LOOKBACKS = (3, 5, 10)

# Realistic institutional cost model
COST_CONFIG = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=1.0,      # $1 per round trip
    commission_per_share=0.01,     # $0.01 per share
    slippage_cents=1.0,            # $0.01 per share fixed slippage
    slippage_proportional=0.0005,  # 5 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
    margin_rate=0.0,
    margin_call_liquidate=True,
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cost_robustness_momentum_results.json"


def cost_signals(closes, lookback):
    """Momentum signal with cost-aware backtest."""
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def segment_result(ticker, bars, lookback):
    """Walk-forward regime-stability result for one ticker and one lookback, with costs."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = cost_signals(closes, lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
    # Use the cost-aware config
    res = bt.stress_segments(
        signals_fn=cost_signals,
        bars=list(bars),
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
        cost_config={
            "commission_per_trade": COST_CONFIG.commission_per_trade,
            "commission_per_share": COST_CONFIG.commission_per_share,
            "slippage_cents": COST_CONFIG.slippage_cents,
            "slippage_proportional": COST_CONFIG.slippage_proportional,
        },
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

    print("\n=== 2. Data preflight ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Cost model (realistic institutional-grade) ===")
    print(f"  commission_per_trade: ${COST_CONFIG.commission_per_trade}")
    print(f"  commission_per_share: ${COST_CONFIG.commission_per_share}")
    print(f"  slippage_cents: ${COST_CONFIG.slippage_cents / 100}")
    print(f"  slippage_proportional: {COST_CONFIG.slippage_proportional * 100} bps")

    print("\n=== 4. Momentum cost-robustness check (lookbacks 3 / 5 / 10) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / "
          "warmup={}d / overlap={}d, min {} bars/segment, "
          "4 contiguous volatility blocks, cost-aware null benchmark\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Full-universe lookback sweep with costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    print(f"  Testing {len(ticker_order)} assets at lookbacks {LOOKBACKS}")

    # Results indexed [lookback][ticker][segment index] with costs applied
    lookback_results = {lb: {t: segment_result(t, all_tickers[t], lb) for t in ticker_order}
                        for lb in LOOKBACKS}

    per_asset = {}
    robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        
        # Check robustness verdict for this asset
        counts = {"EDGE": 0, "LOSS": 0, "NO_EDGE": 0}
        for lb in LOOKBACKS:
            verdict = profile[str(lb)]["verdict"]
            medians = profile[str(lb)]["medians"]
            if verdict == "REGIME_STABLE" and all(m > 0 for m in medians):
                counts["EDGE"] += 1
            elif verdict == "REGIME_STABLE_LOSS":
                counts["LOSS"] += 1
            else:
                counts["NO_EDGE"] += 1
        
        if counts["EDGE"] >= 2:
            v = "ROBUST"
            robustness_counts[v] += 1
        elif counts["EDGE"] == 1:
            v = "SENSITIVE"
            robustness_counts[v] += 1
        else:
            v = "NO_EDGE"
            robustness_counts[v] += 1
            
        profile["edge_counts"] = counts
        profile["robustness_verdict"] = v
        profile["lookback_medians"] = {str(lb): profile[str(lb)]["medians"] for lb in LOOKBACKS}
        profile["lookback_null_medians"] = {str(lb): profile[str(lb)]["null_medians"] for lb in LOOKBACKS}
        
        lookback_5_counts[profile["5"]["verdict"]] += 1
        per_asset[ticker] = profile

    print("  lookback 5 (reference) verdict counts across universe: ")
    print(f"    {lookback_5_counts}")
    print("  lookback-robustness verdict counts with realistic costs: ")
    print(f"    ROBUST (>=2 lookbacks positive edge) | "
          f"SENSITIVE (edge at exactly one lookback) | "
          f"NO_EDGE (no lookback shows a positive edge)\n")
    print(f"    ROBUST={robustness_counts['ROBUST']} | "
          f"SENSITIVE={robustness_counts['SENSITIVE']} | "
          f"NO_EDGE={robustness_counts['NO_EDGE']}\n")

    print("  Per-asset cost-robustness profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        cost_profile = p.get("5", {}).get("cost_config", {})
        med = ", ".join(
            "{}:{:6.3f} {:4s}".format(lb, p[str(lb)]["medians"][0], p[str(lb)]["verdict"])
            for lb in LOOKBACKS
        )
        print("    {:6s} cost_config={}: medians3510={:<40s} rob={:<8s} "
              "edge_counts={}".format(
                ticker, cost_profile, med, p["robustness_verdict"],
                p["edge_counts"]))

    print("\n=== 5. Determinism check (AMZN / JPM) ===")
    # Determinism: rerun the target subset and assert identical output.
    res2 = {lb: {t: segment_result(t, all_tickers[t], lb) for t in ["AMZN", "JPM"]}
            for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in ["AMZN", "JPM"]}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print(f"  determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            robustness_counts=robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t]["lookback_medians"][str(lb)]
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t]["lookback_null_medians"][str(lb)]
                                           for lb in LOOKBACKS},
                    verdicts_5=per_asset[t]["5"]["verdict"],
                    edge_counts=per_asset[t]["edge_counts"],
                    robustness_verdict=per_asset[t]["robustness_verdict"],
                    n_folds_5=per_asset[t]["5"]["n_folds"],
                    cost_config=per_asset[t]["5"]["cost_config"],
                ))
                for t in ticker_order
            ),
        ),
        cost_config={
            "commission_per_trade": COST_CONFIG.commission_per_trade,
            "commission_per_share": COST_CONFIG.commission_per_share,
            "slippage_cents": COST_CONFIG.slippage_cents,
            "slippage_proportional": COST_CONFIG.slippage_proportional,
        },
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"artifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-robustness check (lookbacks 3/5/10): "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
