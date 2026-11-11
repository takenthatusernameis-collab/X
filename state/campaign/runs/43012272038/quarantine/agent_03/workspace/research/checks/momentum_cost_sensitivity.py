"""Real-data transaction-cost sensitivity of short-horizon momentum.

Objective: classify whether the admitted candidate positive evidence
(short-horizon per-asset momentum, `research/checks/momentum.py`, activations
37318950814 / 37361000967) survives the repository's existing realistic
transaction-cost model, and at what cost level it disappears.

Cost model (PREDECLARED, taken verbatim from the repository's own "realistic
costs" configuration, `examples/ma_crossover_real_data.py:131-135` and
`examples/ma_crossover.py:63-70`)::

    commission_per_trade = 2.0        # $2.00 per fill
    commission_per_share = 0.003      # $0.003 per share
    slippage_cents       = 2.0        # 2 cents per share
    slippage_proportional = 0.0005    # 5 bps of fill price

The cost grid is a fixed multiple `k` of that single model, applied uniformly
to both the candidate and the coin-flip null (so the comparison stays matched
to costs):

    k = 0.0   zero-cost reference (must reproduce the existing artifact)
    k = 1.0   the repository's existing realistic cost model
    k = 2.0   conservative 2x stress

Nothing here tunes the signal rule, the universe, the walk-forward windows, the
segmentation, or the null in response to the cost outcome: the grid, the
windows, the verdict thresholds and the classification rule are fixed in this
file before any number is produced.

Signal: long the previous `lookback`-day return, hold 1 day, daily rebalance
(`bt.momentum_signals`), for lookback in {3, 5, 10}. Walk-forward
train=252d / test=84d / warmup=60d / overlap=60d inside the 4 contiguous
volatility blocks of the real-data regime family (`bt.volatility_blocks`,
`bt.stress_segments`), same settings as `momentum_lookback_sweep.py`.

A-priori classification (fixed before running):

Per (asset, lookback), the edge label is the same rule the lookback sweep used
(`momentum_lookback_sweep.lookback_verdict`): EDGE iff the overall verdict is
REGIME_STABLE and every segment median is > 0.

Per asset, over lookbacks {3, 5, 10}:
    COST_ROBUST     : at least one lookback keeps a positive EDGE at k = 2.0
    COST_FRAGILE    : at least one lookback keeps a positive EDGE at k = 1.0,
                      but no lookback keeps one at k = 2.0
    COST_ELIMINATED : no lookback keeps a positive EDGE at k = 1.0

Interpretation guard: a COST_FRAGILE or COST_ELIMINATED verdict is only
evidence about execution cost, not about the signal's information content;
costs are a design constraint, not a falsification of return predictability.

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

# The repository's existing realistic cost model, verbatim.
BASE_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
COST_LEVELS = (0.0, 1.0, 2.0)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
SWEEP_ARTIFACT_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def cost_cfg(multiple):
    """BacktestConfig for `multiple` x the repository's realistic cost model."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        commission_per_trade=BASE_COST["commission_per_trade"] * multiple,
        commission_per_share=BASE_COST["commission_per_share"] * multiple,
        slippage_cents=BASE_COST["slippage_cents"] * multiple,
        slippage_proportional=BASE_COST["slippage_proportional"] * multiple,
    )


def segment_result(bars, lookback, multiple):
    """Walk-forward regime-stability result for one asset and one cost level."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"signals {len(sig)} != bars {len(closes)}")
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
        cfg=cost_cfg(multiple),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def edge_label(sr):
    """EDGE iff REGIME_STABLE with every segment median strictly positive."""
    if sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"]):
        return "EDGE"
    if sr["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def classify(levels_for_asset):
    """Per-asset cost-robustness class from the per-level edge labels.

    `levels_for_asset` maps lookback -> cost-level key -> segment result dict.
    """
    def has_edge(level_key):
        return any(
            edge_label(levels_for_asset[str(lb)][level_key]) == "EDGE"
            for lb in LOOKBACKS
        )

    if has_edge("2.0"):
        return "COST_ROBUST"
    if has_edge("1.0"):
        return "COST_FRAGILE"
    return "COST_ELIMINATED"


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print(f"  {entry['ticker']}: "
              f"{'OK' if actual == entry['checksum_sha256'] else 'MISMATCH'}")
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

    print("\n=== 3. Leakage + fill-equity gate (AMZN, JPM; realistic cost) ===")
    tickers = {}
    for ticker in ("AMZN", "JPM"):
        bars = bt.load_ticker(ticker)[0]
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cost_cfg(1.0))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        n_fills = len(res.trades)
        total_cost = float(sum(f.commission for f in res.trades))
        print(f"  {ticker}: {bars.n_bars} bars, fills={n_fills}, "
              f"cost=${total_cost:,.0f} "
              f"({total_cost / 1e6:.2%} of initial capital), leakage [PASS]")
    print("  no-look-ahead: the momentum sign at bar t uses closes[:t] only; "
          "costs apply to every round trip at the same fill price.")

    print("\n=== 4. Cost grid (multiples of the repository's realistic model) ===")
    print(f"  base model: commission_per_trade={BASE_COST['commission_per_trade']}, "
          f"commission_per_share={BASE_COST['commission_per_share']}, "
          f"slippage_cents={BASE_COST['slippage_cents']}, "
          f"slippage_proportional={BASE_COST['slippage_proportional']}")
    print(f"  levels: {list(COST_LEVELS)} | lookbacks: {list(LOOKBACKS)} | "
          f"train={TRAIN}d test={TEST}d warmup={WARM}d overlap={OVERLAP}d | "
          f"min {MIN_SEGMENT_BARS} bars/segment | {N_BLOCKS} volatility blocks\n")

    ticker_order = [e["ticker"] for e in manifest["entries"]]
    all_tickers = dict((t, bt.load_ticker(t)[0]) for t in ticker_order)

    # results[ticker][lookback][cost_level_key] -> segment result dict
    results = {}
    for ticker in ticker_order:
        results[ticker] = {}
        for lb in LOOKBACKS:
            results[ticker][str(lb)] = {}
            for k in COST_LEVELS:
                results[ticker][str(lb)][str(k)] = segment_result(
                    all_tickers[ticker], lb, k
                )
        print(f"  {ticker}: done")

    print("\n  Per-asset per-level edge labels (EDGE / NO_EDGE / LOSS):")
    for ticker in ticker_order:
        cells = []
        for lb in LOOKBACKS:
            for k in COST_LEVELS:
                sr = results[ticker][str(lb)][str(k)]
                cells.append(f"lb{lb}@{k}x={edge_label(sr)}")
        print(f"    {ticker:6s} " + "  ".join(cells))

    per_asset = {}
    class_counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "COST_ELIMINATED": 0}
    level_edge_counts = dict(
        (str(k), {"EDGE": 0, "NO_EDGE": 0, "LOSS": 0}) for k in COST_LEVELS
    )
    for ticker in ticker_order:
        entry = {"by_level": {}}
        for lb in LOOKBACKS:
            for k in COST_LEVELS:
                sr = results[ticker][str(lb)][str(k)]
                entry["by_level"].setdefault(str(lb), {})[str(k)] = sr
                level_edge_counts[str(k)][edge_label(sr)] += 1
        cls = classify(entry["by_level"])
        entry["cost_class"] = cls
        class_counts[cls] += 1
        per_asset[ticker] = entry

    print("\n  Edge-cell counts per cost level across the universe "
          f"({len(ticker_order)} assets x {len(LOOKBACKS)} lookbacks):")
    for k in COST_LEVELS:
        c = level_edge_counts[str(k)]
        print(f"    k={k}: EDGE={c['EDGE']} NO_EDGE={c['NO_EDGE']} LOSS={c['LOSS']}")
    print(f"\n  Cost-robustness classes: {class_counts}")

    # Determinism: rerun the AMZN/JPM subset at all levels and compare.
    res2 = {}
    for ticker in ("AMZN", "JPM"):
        res2[ticker] = {}
        for lb in LOOKBACKS:
            res2[ticker][str(lb)] = {}
            for k in COST_LEVELS:
                res2[ticker][str(lb)][str(k)] = segment_result(
                    tickers[ticker], lb, k
                )
    out1 = json.dumps(
        dict((t, results[t][str(lb)][str(k)])
             for t in ("AMZN", "JPM") for lb in LOOKBACKS for k in COST_LEVELS),
        sort_keys=True,
    )
    out2 = json.dumps(
        dict((t, res2[t][str(lb)][str(k)])
             for t in ("AMZN", "JPM") for lb in LOOKBACKS for k in COST_LEVELS),
        sort_keys=True,
    )
    print("\ndeterminism: AMZN/JPM all lookbacks x all levels r1 == r2: "
          f"{out1 == out2}")
    assert out1 == out2, "non-deterministic output"

    # Cross-consistency: k = 0 must reproduce the existing zero-cost artifact.
    cross = {"checked": 0, "mismatches": []}
    if SWEEP_ARTIFACT_PATH.exists():
        with open(SWEEP_ARTIFACT_PATH) as f:
            sweep = json.load(f)
        for ticker in ticker_order:
            for lb in LOOKBACKS:
                pub = sweep["per_asset"][ticker][str(lb)]
                mine = results[ticker][str(lb)]["0.0"]
                cross["checked"] += 1
                if (mine["medians"] != pub["medians"]
                        or mine["null_medians"] != pub["null_medians"]
                        or mine["verdict"] != pub["verdict"]):
                    cross["mismatches"].append(
                        dict(ticker=ticker, lookback=lb,
                             published_medians=pub["medians"],
                             recomputed_medians=mine["medians"],
                             published_verdict=pub["verdict"],
                             recomputed_verdict=mine["verdict"])
                    )
        print(f"k=0 cross-consistency vs momentum_lookback_sweep_results.json: "
              f"{cross['checked'] - len(cross['mismatches'])}/{cross['checked']} MATCH")
        for mm in cross["mismatches"]:
            print(f"  MISMATCH {mm}")
    else:
        print("k=0 cross-consistency: artifact not present; gate not evaluated")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_levels=list(COST_LEVELS),
        base_cost_model=BASE_COST,
        cost_model_source="examples/ma_crossover_real_data.py:131-135",
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        per_asset=per_asset,
        universe=dict(
            ticker_order=ticker_order,
            cost_class_counts=class_counts,
            level_edge_counts=level_edge_counts,
            zero_cost_cross_consistency=cross,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity on the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())