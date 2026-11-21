"""Real-data transaction-cost sensitivity of the short-horizon momentum class.

Objective (task R-001): decide whether the admitted candidate positive evidence
(short-horizon momentum, lookback 3/5/10, hold 1 day, daily rebalance) survives
conservative transaction-cost stress on the collected 10-asset universe, judged
against a cost-matched coin-flip null.

Why this is the decision-relevant qualification: the momentum evidence
(`momentum.py`, `momentum_lookback_sweep.py`, `momentum_longer_horizon.py`) was
produced with the framework's default zero-cost `BacktestConfig`. The rule is a
daily-rebalanced +/-1 exposure, so its turnover is close to 2x notional per bar;
the repo's realistic cost model (`examples/ma_crossover_real_data.py`) charges
commission per trade + per share, fixed slippage in cents per share, and
proportional slippage. Whether the ~+0.05..+0.13 per-fold medians are a gross
artifact or a net edge is therefore unresolved by the existing artifacts.

PREDECLARED COST GRID (fixed before execution; not tuned to results):
  zero         : BacktestConfig()  (replicates the existing zero-cost path)
  realistic    : commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005
                 (the repository's existing realistic model)
  conservative : commission_per_trade=5.0, commission_per_share=0.005,
                 slippage_cents=5.0, slippage_proportional=0.0015
                 (~2.5x-3x the realistic model)

PREDECLARED CLASSIFICATION (fixed before execution):
  edge(level)   = verdict(level) == "REGIME_STABLE" and every segment median > 0
                  (the same EDGE rule used by momentum_lookback_sweep.py)
  cost_class(asset, lookback):
    COST_ROBUST   : edge at zero, realistic and conservative
    COST_SENSITIVE: edge at zero and realistic, lost at conservative
    COST_FRAGILE  : edge at zero, lost at realistic
    NO_EDGE       : no edge at zero
  universe_verdict (evaluated at lookback 5, the admitted parameterization):
    COST_ROBUST   : >= 5/10 assets COST_ROBUST
    MIXED         : >= 1 COST_ROBUST or COST_SENSITIVE, but < 5 COST_ROBUST
    COST_FRAGILE  : 0 COST_ROBUST/COST_SENSITIVE and >= 1 COST_FRAGILE
    NO_EDGE       : no asset had a zero-cost edge

A-priori prediction (recorded before execution, falsifiable): because the rule
is a high-turnover daily-rebalanced exposure, realistic costs are expected to
remove a material part of the gross edge; the universe classification is
expected to be MIXED or COST_FRAGILE rather than COST_ROBUST. A COST_ROBUST
outcome would refute the concern that the momentum candidate is a gross-of-cost
artifact. Either outcome is recorded; the classification is not tuned afterward.

Method: seed 42; dataset yf-ohlcv-universe-2009-to-2026-10-03; the same
walk-forward regime-stability gate as the momentum cell (train=252d/test=84d/
warmup=60d/overlap=60d, 4 contiguous volatility blocks, min 400 bars/segment),
run per lookback in {3, 5, 10} and per cost level, with `stress_segments`
receiving the level's `BacktestConfig` so both the candidate and the framework's
coin-flip null are charged the same costs. The zero-cost column is cross-checked
against `momentum_lookback_sweep_results.json`, and determinism is asserted by
an independent rerun of the AMZN/JPM subset.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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
REFERENCE_LOOKBACK = 5

COST_LEVELS = (
    ("zero", {}),
    ("realistic", dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )),
    ("conservative", dict(
        commission_per_trade=5.0,
        commission_per_share=0.005,
        slippage_cents=5.0,
        slippage_proportional=0.0015,
    )),
)

COST_MODEL_FIELDS = (
    "commission_per_trade",
    "commission_per_share",
    "slippage_cents",
    "slippage_proportional",
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
LOOKBACK_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def cost_config(level: str) -> bt.BacktestConfig:
    """BacktestConfig for one predeclared cost level."""
    params = dict(COST_LEVELS)[level]
    return bt.BacktestConfig(**params)


def segment_result(ticker, bars, lookback, cfg) -> dict:
    """Walk-forward regime-stability result for one ticker and lookback at one
    cost level, with the coin-flip null charged the same costs."""
    closes = bars.closes_array()
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
        cfg=cfg,
    )
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 6) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 6) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 6),
        null_dispersion=round(res.null_dispersion, 6),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def edge_present(sr: dict) -> bool:
    """The momentum EDGE rule: REGIME_STABLE with uniformly positive medians."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def classify_cost_profile(profile: dict) -> str:
    """Predeclared per-(asset, lookback) cost classification."""
    edges = {level: edge_present(profile[level]) for level, _ in COST_LEVELS}
    if not edges["zero"]:
        return "NO_EDGE"
    if edges["realistic"] and edges["conservative"]:
        return "COST_ROBUST"
    if edges["realistic"]:
        return "COST_SENSITIVE"
    return "COST_FRAGILE"


def universe_verdict(classes) -> str:
    """Predeclared universe-level classification from per-asset cost classes."""
    counts = Counter(classes)
    robust = counts["COST_ROBUST"]
    sensitive = counts["COST_SENSITIVE"]
    fragile = counts["COST_FRAGILE"]
    if robust >= 5:
        return "COST_ROBUST"
    if robust >= 1 or sensitive >= 1:
        return "MIXED"
    if fragile >= 1:
        return "COST_FRAGILE"
    return "NO_EDGE"


def retention(profile: dict, level: str):
    """Median realistic/conservative segment median divided by the zero-cost
    median, for assets with a positive zero-cost median (None otherwise)."""
    zero_med = float(np.median(profile["zero"]["medians"]))
    if zero_med <= 0:
        return None
    level_med = float(np.median(profile[level]["medians"]))
    return round(level_med / zero_med, 4)


def leakage_review(tickers, cfg) -> None:
    """Signal-integrity and equity-fill audit with the realistic cost model."""
    for ticker in tickers:
        bars, dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cfg)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, cfg.initial_capital)
        total_return = (res.equity_curve[-1] - cfg.initial_capital) / cfg.initial_capital
        print(f"  {ticker}: {bars.n_bars} bars, signals {len(signals)}, "
              f"leakage [PASS], realistic-cost total_return {total_return:+.3f}")


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

    print("\n=== 3. Leakage review (AMZN, JPM; realistic cost model) ===")
    target = ["AMZN", "JPM"]
    leakage_review(target, cost_config("realistic"))
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only.")

    print("\n=== 4. Cost grid (predeclared) ===")
    for name, params in COST_LEVELS:
        filled = {f: params.get(f, 0.0) for f in COST_MODEL_FIELDS}
        print(f"  {name:12s}: {filled}")
    print("  each cost level is applied to the candidate AND the coin-flip null "
          "on the same segments\n")

    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(tickers.keys())

    per_asset = {}
    for ticker in ticker_order:
        bars = tickers[ticker]
        per_asset[ticker] = {}
        for lb in LOOKBACKS:
            profile = {}
            for level, _ in COST_LEVELS:
                profile[level] = segment_result(ticker, bars, lb, cost_config(level))
            entry = dict(
                levels=profile,
                edges={level: edge_present(profile[level]) for level, _ in COST_LEVELS},
                cost_class=classify_cost_profile(profile),
                realistic_retention=retention(profile, "realistic"),
                conservative_retention=retention(profile, "conservative"),
                realistic_minus_zero=round(
                    float(np.median(profile["realistic"]["medians"]))
                    - float(np.median(profile["zero"]["medians"])), 6),
                conservative_minus_zero=round(
                    float(np.median(profile["conservative"]["medians"]))
                    - float(np.median(profile["zero"]["medians"])), 6),
            )
            per_asset[ticker][str(lb)] = entry
        classes = [per_asset[ticker][str(lb)]["cost_class"] for lb in LOOKBACKS]
        print("  {:6s} zero/real/cp edge {} | classes {} | retention "
              "real={} cons={}".format(
                  ticker,
                  "".join("1" if per_asset[ticker][str(lb)]["edges"]["realistic"] else "0"
                          for lb in LOOKBACKS),
                  classes,
                  [per_asset[ticker][str(lb)]["realistic_retention"] for lb in LOOKBACKS],
                  [per_asset[ticker][str(lb)]["conservative_retention"] for lb in LOOKBACKS]))
    print("")

    print("=== 5. Cost classification summary ===")
    summary = dict(
        cost_class_counts={},
        per_lookback=dict(),
        universe_verdict_by_lookback={},
        universe_verdict=None,
    )
    for lb in LOOKBACKS:
        classes = [per_asset[t]["{}".format(lb)]["cost_class"] for t in ticker_order]
        counts = Counter(classes)
        robust_assets = [t for t in ticker_order
                         if per_asset[t][str(lb)]["cost_class"] == "COST_ROBUST"]
        uverd = universe_verdict(classes)
        summary["per_lookback"][str(lb)] = dict(
            cost_class_counts=dict(counts),
            cost_robust_assets=robust_assets,
            universe_verdict=uverd,
        )
        summary["universe_verdict_by_lookback"][str(lb)] = uverd
        print("  lookback {:2d}: {} | COST_ROBUST assets {}/{} -> {}".format(
            lb, dict(counts), len(robust_assets), len(ticker_order), uverd))

    all_classes = [per_asset[t][str(lb)]["cost_class"]
                   for t in ticker_order for lb in LOOKBACKS]
    summary["cost_class_counts"] = dict(Counter(all_classes))
    summary["universe_verdict"] = summary["universe_verdict_by_lookback"][
        str(REFERENCE_LOOKBACK)]
    print("  all (asset, lookback) cells: {}".format(dict(Counter(all_classes))))
    print("  overall universe verdict (lookback {}): {}".format(
        REFERENCE_LOOKBACK, summary["universe_verdict"]))

    print("\n=== 6. Zero-cost cross-consistency vs momentum_lookback_sweep_results.json ===")
    with open(LOOKBACK_ARTIFACT) as f:
        prior = json.load(f)
    cmismatch = False
    for ticker in ticker_order:
        for lb in LOOKBACKS:
            new = per_asset[ticker][str(lb)]["levels"]["zero"]
            old = prior["per_asset"][ticker][str(lb)]
            ok = (
                [round(v, 3) for v in new["medians"]] == [round(v, 3) for v in old["medians"]]
                and [round(v, 3) for v in new["null_medians"]] == [round(v, 3) for v in old["null_medians"]]
                and new["segments"] == old["segments"]
                and new["verdict"] == old["verdict"]
                and new["n_folds"] == old["n_folds"]
            )
            if not ok:
                cmismatch = True
                print("  MISMATCH {} lookback {}: new {} vs old {}".format(
                    ticker, lb, new, old))
    if cmismatch:
        print("ZERO-COST CROSS-CONSISTENCY FAILED; not writing artifact.")
        sys.exit(1)
    print("  all 10 tickers x 3 lookbacks reproduce the prior zero-cost artifact -> MATCH")

    print("\n=== 7. Determinism (AMZN, JPM; all lookbacks and cost levels) ===")
    det_ok = True
    for ticker in target:
        bars = tickers[ticker]
        for lb in LOOKBACKS:
            for level, _ in COST_LEVELS:
                r1 = segment_result(ticker, bars, lb, cost_config(level))
                r2 = segment_result(ticker, bars, lb, cost_config(level))
                if json.dumps(r1, sort_keys=True) != json.dumps(r2, sort_keys=True):
                    det_ok = False
                    print("  NON-DETERMINISTIC {} lookback {} level {}".format(
                        ticker, lb, level))
    print("  determinism r1 == r2: {}".format(det_ok))
    if not det_ok:
        print("DETERMINISM FAILED; not writing artifact.")
        sys.exit(1)

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        reference_lookback=REFERENCE_LOOKBACK,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_levels={name: {f: dict(params).get(f, 0.0) for f in COST_MODEL_FIELDS}
                     for name, params in COST_LEVELS},
        cost_model_source="examples/ma_crossover_real_data.py (realistic level)",
        cost_level_order=[name for name, _ in COST_LEVELS],
        classification_rule=(
            "edge(level) = REGIME_STABLE and all segment medians > 0; "
            "COST_ROBUST = edge at zero+realistic+conservative; "
            "COST_SENSITIVE = edge at zero+realistic, lost at conservative; "
            "COST_FRAGILE = edge at zero, lost at realistic; "
            "NO_EDGE = no zero-cost edge; "
            "universe COST_ROBUST >= 5/10, MIXED >= 1 robust/sensitive, "
            "COST_FRAGILE 0 robust/sensitive and >= 1 fragile, else NO_EDGE"
        ),
        a_priori_prediction=(
            "High-turnover daily-rebalanced +/-1 exposure: realistic costs are "
            "expected to remove a material part of the gross edge, so the "
            "universe verdict is expected to be MIXED or COST_FRAGILE rather "
            "than COST_ROBUST; COST_ROBUST would refute the gross-artifact "
            "concern."
        ),
        per_asset=per_asset,
        summary=summary,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity across lookback 3/5/10 on the "
          "collected universe: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())