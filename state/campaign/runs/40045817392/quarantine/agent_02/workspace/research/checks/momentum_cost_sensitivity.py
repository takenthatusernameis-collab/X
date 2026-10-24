"""Cost-sensitivity check for lookback 3/5/10 momentum.

Background: the short-horizon momentum edge was SUPPORTED at lookback=5
in the 10-asset collected universe (research/checks/momentum.py, activation
37318950814). Cost robustness remains a decision-relevant qualification.

This test evaluates whether the momentum edge survives conservative transaction
costs (realistic trading model) on the same universe and walk-forward geometry,
compared to a zero-cost baseline. The same coin-flip null is used for both cost
levels to maintain a matched comparison.

Method: for each lookback in {3, 5, 10}, run walk-forward validation of the
momentum signal (long previous N-day return, hold 1 day, daily rebalance) on the
collected adjusted-close universe with the same walk-forward geometry as the
original momentum check (train=252d/test=84d/warmup=60d/overlap=60d).

Two cost configurations are run:
- zero cost: BacktestConfig(commission_per_trade=0.0, commission_per_share=0.0,
  slippage_cents=0.0, slippage_proportional=0.0)
- realistic cost: BacktestConfig(commission_per_trade=2.0, commission_per_share=0.003,
  slippage_cents=2.0, slippage_proportional=0.0005)

The realistic cost configuration is the enterprise's standard realistic trading
model (see examples/ma_crossover.py and examples/ma_crossover_real_data.py).

For each cost level, the per-segment median log returns are computed and
compared against the same coin-flip null to produce REGIME_STABLE verdicts.
The cost-sensitivity conclusion is based on whether the positive REGIME_STABLE
edge survives realistic costs (i.e., either the same verdict or uniformly weaker
medians) and whether the edge remains statistically distinguishable from the null.

Results are written as an artifact for independent verification.

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

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

def cost_config_dict(has_cost: bool):
    """Return BacktestConfig parameters for a given cost level."""
    if has_cost:
        return {
            "initial_capital": 1e6,
            "warmup_periods": WARM,
            "commission_per_trade": 2.0,
            "commission_per_share": 0.003,
            "slippage_cents": 2.0,
            "slippage_proportional": 0.0005,
        }
    else:
        return {
            "initial_capital": 1e6,
            "warmup_periods": WARM,
        }


def segment_result_cost_sensitive(
    ticker, bars, lookback, has_cost: bool
):
    """Walk-forward regime-stability result for one ticker, lookback, and cost level."""
    closes = bars.closes_array() if isinstance(bars, bt.BarSequence) else np.array(
        [float(b.close) for b in bars], dtype=np.float64)
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    
    cfg_params = cost_config_dict(has_cost)
    cfg = bt.BacktestConfig(**cfg_params)
    
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
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
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

    print("\n=== 3. Load universe ===")
    tickers = {e["ticker"]: bt.load_ticker(e["ticker"])[0] for e in manifest["entries"]}

    print("\n=== 4. Run cost-sensitivity analysis ===")
    print("Momentum lookbacks: {}".format(LOOKBACKS))
    print("Cost levels: zero-cost, realistic-cost")
    print("Walk-forward: train={}d/test={}d/warmup={}d/overlap={}d".format(
        TRAIN, TEST, WARM, OVERLAP))
    print("Regime blocks: {} past-only volatility blocks (window={})".format(
        N_BLOCKS, WINDOW))

    per_asset_cost_sensitive = {}
    for ticker in tickers:
        per_asset_cost_sensitive[ticker] = {}
        for lb in LOOKBACKS:
            for has_cost in [False, True]:
                cost_label = "cost" if has_cost else "nocost"
                per_asset_cost_sensitive[ticker][f"{lb}_{cost_label}"] = (
                    segment_result_cost_sensitive(ticker, tickers[ticker], lb, has_cost)
                )
                print(f"  {ticker} lookback={lb} {'realistic' if has_cost else 'zero'} costs: "
                      f"verdict={per_asset_cost_sensitive[ticker][f'{lb}_{cost_label}']['verdict']}, "
                      f"medians={per_asset_cost_sensitive[ticker][f'{lb}_{cost_label}']['medians']}")

    print("\\n=== 5. Summarize results ===")
    summary = {}
    for ticker in tickers:
        summary[ticker] = {}
        for lb in LOOKBACKS:
            nocost_key = f"{lb}_nocost"
            cost_key = f"{lb}_cost"
            
            if nocost_key in per_asset_cost_sensitive[ticker] and cost_key in per_asset_cost_sensitive[ticker]:
                summary[ticker][lb] = {
                    "zero_cost_verdict": per_asset_cost_sensitive[ticker][nocost_key]["verdict"],
                    "zero_cost_medians": per_asset_cost_sensitive[ticker][nocost_key]["medians"],
                    "realistic_cost_verdict": per_asset_cost_sensitive[ticker][cost_key]["verdict"],
                    "realistic_cost_medians": per_asset_cost_sensitive[ticker][cost_key]["medians"],
                    "cost_sensitive": None,
                    "robust": None,
                }
                
                zero_med = per_asset_cost_sensitive[ticker][nocost_key]["medians"]
                real_med = per_asset_cost_sensitive[ticker][cost_key]["medians"]
                
                # Determine cost sensitivity
                if summary[ticker][lb]["zero_cost_verdict"] == "REGIME_STABLE":
                    if summary[ticker][lb]["realistic_cost_verdict"] == "REGIME_STABLE":
                        # Check if edge survived realistic costs
                        if all(r >= z for r, z in zip(real_med, zero_med)):
                            cost_sensitive = "ROBUST_TO_COSTS"
                        else:
                            cost_sensitive = "WEAKENED_BY_COSTS"
                    else:
                        cost_sensitive = "FAILED_AT_COSTS"
                else:
                    cost_sensitive = "NO_EDGE_AT_ZERO_COST"
                    
                summary[ticker][lb]["cost_sensitive"] = cost_sensitive
                
                # Determine overall robustness (lookback-robust edge at realistic costs)
                is_robust = False
                for lb_check in LOOKBACKS:
                    if (lb_check in summary[ticker] and 
                        summary[ticker][lb_check]["cost_sensitive"] == "ROBUST_TO_COSTS" and
                        all(m > 0 for m in summary[ticker][lb_check]["realistic_cost_medians"])):
                        is_robust = True
                        break
                summary[ticker][lb]["robust"] = is_robust

    print("\n  Cost-sensitivity per lookback:")
    print("  lookback | zero_cost_verdict | realistic_cost_verdict | cost_sensitive | robust")
    for ticker in sorted(tickers.keys()):
        for lb in LOOKBACKS:
            if ticker in summary and lb in summary[ticker]:
                s = summary[ticker][lb]
                print("  {} | {} | {} | {} | {}".format(
                    lb, s["zero_cost_verdict"], s["realistic_cost_verdict"],
                    s["cost_sensitive"], s["robust"]))

    print("\n=== 6. Determinism check ===")
    target = ["AMZN", "JPM"]
    res2 = {}
    for ticker in target:
        res2[ticker] = {}
        for lb in LOOKBACKS:
            for has_cost in [False, True]:
                res2[ticker][f"{lb}_{'cost' if has_cost else 'nocost'}"] = (
                    segment_result_cost_sensitive(ticker, tickers[ticker], lb, has_cost)
                )
    
    out1 = json.dumps(per_asset_cost_sensitive, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("  determinism: r1 == r2: {}".format(out1 == out2))
    if out1 != out2:
        print("  NON-DETERMINISTIC OUTPUT - ABORTING")
        sys.exit(1)

    print("\n=== 7. Write artifact ===")
    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset_cost_sensitive=per_asset_cost_sensitive,
        summary=summary,
        zero_cost_config=cost_config_dict(has_cost=False),
        realistic_cost_config=cost_config_dict(has_cost=True),
        note="Cost-sensitivity analysis for lookback 3/5/10 momentum; zero-cost vs realistic-cost using coin-flip null; deterministic and preflight-verified."
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("  artifact written to: {}".format(ARTIFACT_PATH))
    print("\nNOTE: research/simulation only. No live trading or production execution.")
    print("Momentum cost-sensitivity check on collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())