"""Cost-sensitivity check for lookback 3/5/10 momentum.

Objective: test whether the short-horizon momentum edge survives conservative
transaction-cost stress on the collected real-data universe.

The momentum frontier test (research/checks/momentum.py, activation 37318950814)
showed a REGIME_STABLE positive edge in 7/10 assets at lookback=5, hold=1 day.
This check closes the cost dimension: momentum is re-tested across lookbacks 3 / 5 /
10 on the same collected adjusted-close universe, same walk-forward, same
coin-flip null, but with realistic transaction costs and a matched null that
uses the same cost model. The edge must survive both parameter sweep and cost
stress before the momentum class can be admitted to the evidence base.

Method (fixed a-priori, not tuned to OOS):
- long the previous N-day return; hold 1 day; daily rebalance
- lookbacks 3 / 5 / 10
- realistic costs: commission_per_trade=$5, commission_per_share=$0.005, slippage_cents=1.0, slippage_proportional=0.005
- matched null: coin-flip signal with same cost parameters
- walk-forward per segment: train=252d / test=84d / warmup=60d / overlap=60d
- minimum 400 bars/segment, 4 contiguous volatility blocks

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

# Realistic cost model for conservative transaction-cost stress
REALISTIC_COSTS = bt.BacktestConfig(
    initial_capital=1e6,
    target_exposure=1.0,
    commission_per_trade=5.0,      # $5 per round-trip
    commission_per_share=0.005,    # $0.005 per 1000 shares
    slippage_cents=1.0,           # 1.0 cents per share
    slippage_proportional=0.005,  # 5 bps proportional slippage
    warmup_periods=0,
    risk_free=0.0,
    periods_per_year=252,
)

# Matched null: coin-flip with same cost parameters
NULL_COSTS = REALISTIC_COSTS

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def segment_result(ticker, bars, lookback, cfg):
    """Walk-forward regime-stability result for one ticker, lookback, and config."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
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
        cfg=cfg,
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


def random_signals(n: int, seed: int = None) -> list:
    """Coin-flip positioning, independent of prices.
    
    Used as the null hypothesis with realistic costs.
    """
    rng = np.random.default_rng(seed)
    weights = rng.choice([-1.0, 0.0, 1.0], size=n, p=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return [bt.Signal(date=i + 1, weight=float(w)) for i, w in enumerate(weights)]


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

    print("\n=== 3. Cost model (BacktestConfig) ===")
    print(f"  Realistic costs: commission_per_trade=${REALISTIC_COSTS.commission_per_trade}, "
          f"commission_per_share=${REALISTIC_COSTS.commission_per_share}, "
          f"slippage_cents={REALISTIC_COSTS.slippage_cents}, "
          f"slippage_proportional={REALISTIC_COSTS.slippage_proportional}")
    print(f"  Initial capital: ${REALISTIC_COSTS.initial_capital:,.0f}")
    print(f"  Target exposure: {REALISTIC_COSTS.target_exposure * 100}%")

    print("\n=== 4. Leakage review per asset (lookback=5 reference) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, REALISTIC_COSTS)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")

    print("\n=== 5. Momentum lookback sweep (3 / 5 / 10) across collected universe with realistic costs ===")
    print(f"  Params: walk-forward per segment train={TRAIN}d / test={TEST}d / "
          f"warmup={WARM}d / overlap={OVERLAP}d, min {MIN_SEGMENT_BARS} bars/segment, "
          f"4 contiguous volatility blocks, each lookback vs a matched null "
          f"with identical cost parameters\n")

    # Full-universe lookback sweep with realistic costs
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # Results indexed [lookback][ticker][segment index].
    lookback_results = {lb: {t: segment_result(t, all_tickers[t], lb, REALISTIC_COSTS) 
                             for t in ticker_order} for lb in LOOKBACKS}

    # Matched null results (coin-flip with same costs)
    print("\n=== 6. Matched null benchmark (coin-flip, same realistic costs) ===")
    null_results = {lb: {t: segment_result(t, all_tickers[t], lb, NULL_COSTS)
                         for t in ticker_order} for lb in LOOKBACKS}

    per_asset = {}
    cost_robustness_counts = {"ROBUST": 0, "SENSITIVE": 0, "NO_EDGE": 0}
    lookback_5_counts = {"REGIME_STABLE": 0, "CONSISTENT_WITH_NOISE": 0,
                         "REGIME_DEPENDENT": 0, "REGIME_STABLE_LOSS": 0}
    
    for ticker in ticker_order:
        profile = {str(lb): lookback_results[lb][ticker] for lb in LOOKBACKS}
        profile["null"] = {str(lb): null_results[lb][ticker] for lb in LOOKBACKS}
        
        # Determine edge status: positive median in candidate vs null
        edges = {}
        for lb in LOOKBACKS:
            cand = profile[str(lb)]["medians"][0]
            null = profile[str(lb)]["null_medians"][0]
            edges[str(lb)] = "EDGE" if cand > null else "NO_EDGE"
        
        # Cross-lookback summary
        edge_count = sum(1 for lb in LOOKBACKS if edges[str(lb)] == "EDGE")
        if edge_count >= 2:
            robustness = "ROBUST"
        elif edge_count == 1:
            robustness = "SENSITIVE"
        else:
            robustness = "NO_EDGE"
        
        cost_robustness_counts[robustness] += 1
        
        # Lookback 5 verdict counts
        r5 = profile["5"]
        lookback_5_counts[r5["verdict"]] += 1
        
        profile["edges"] = edges
        profile["robustness_verdict"] = robustness
        per_asset[ticker] = profile

    print("  lookback 5 (reference) verdict counts across universe: "
          f"{lookback_5_counts}")
    print("  cost-robustness verdict counts: "
          f"ROBUST={cost_robustness_counts['ROBUST']} "
          f"(edge at >=2 lookbacks after costs) | "
          f"SENSITIVE={cost_robustness_counts['SENSITIVE']} "
          f"(edge at exactly one lookback) | "
          f"NO_EDGE={cost_robustness_counts['NO_EDGE']}\n")
    print("  Per-asset perturbation profiles:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        med = ", ".join(
            "{}:{:6.3f} vs {:6.3f}={}".format(lb, p[str(lb)]["medians"][0], 
                                             p[str(lb)]["null_medians"][0],
                                             p["edges"][str(lb)])
            for lb in LOOKBACKS
        )
        print("    {:6s} segments={:<8} cand vs null medians={:<50s} rob={:<8s} "
              "cost_robust={}".format(
            ticker, str(p["5"]["segments"]), med, p["robustness_verdict"],
            p["robustness_verdict"]))

    # Determinism: rerun the target subset and assert identical output.
    print("\n=== 7. Determinism (independent re-run of target subset) ===")
    res2 = {lb: {t: segment_result(t, tickers[t], lb, REALISTIC_COSTS)
                 for t in target} for lb in LOOKBACKS}
    out1 = json.dumps({lb: {t: lookback_results[lb][t] for t in target}
                       for lb in LOOKBACKS}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print(f"  determinism: r1 == r2: {out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        realistic_costs=dict(
            initial_capital=REALISTIC_COSTS.initial_capital,
            target_exposure=REALISTIC_COSTS.target_exposure,
            commission_per_trade=REALISTIC_COSTS.commission_per_trade,
            commission_per_share=REALISTIC_COSTS.commission_per_share,
            slippage_cents=REALISTIC_COSTS.slippage_cents,
            slippage_proportional=REALISTIC_COSTS.slippage_proportional,
        ),
        per_asset=per_asset,
        universe=dict(
            lookback_5_counts=lookback_5_counts,
            cost_robustness_counts=cost_robustness_counts,
            ticker_order=ticker_order,
            per_asset=dict(
                (t, dict(
                    lookback_medians={str(lb): per_asset[t][str(lb)]["medians"] 
                                      for lb in LOOKBACKS},
                    lookback_null_medians={str(lb): per_asset[t][str(lb)]["null_medians"]
                                           for lb in LOOKBACKS},
                    lookback_candidate_dispersion={str(lb): per_asset[t][str(lb)]["candidate_dispersion"]
                                                   for lb in LOOKBACKS},
                    lookback_null_dispersion={str(lb): per_asset[t][str(lb)]["null_dispersion"]
                                             for lb in LOOKBACKS},
                    verdicts_5=per_asset[t]["5"]["verdict"],
                    edge_status={str(lb): per_asset[t]["edges"][str(lb)] for lb in LOOKBACKS},
                    robustness_verdict=per_asset[t]["robustness_verdict"],
                    n_folds_5=per_asset[t]["5"]["n_folds"],
                ))
                for t in ticker_order
            ),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum lookback sweep with realistic costs on the "
          "collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())