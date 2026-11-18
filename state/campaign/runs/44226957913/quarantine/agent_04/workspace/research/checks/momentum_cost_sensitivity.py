"""Bounded transaction-cost sensitivity of short-horizon momentum.

Objective (campaign task R-001): the momentum class was admitted to the
evidence base as candidate *positive* evidence (activations 37318950814 and
37361000967), but every momentum run used `BacktestConfig(initial_capital=1e6)`
with the default zero-cost fields -- i.e. the admitted edge is a *gross*
edge. Momentum is a mechanical, daily-rebalanced direction bet, so its
evidence is only decision-relevant if it survives the repository's existing
realistic transaction-cost model.

This check re-runs lookback 3 / 5 / 10 momentum through the same
walk-forward regime-stability gate (train=252d / test=84d / warmup=60d /
overlap=60d, 4 contiguous volatility blocks, min 400 bars/segment, each
lookback compared against a coin-flip sign null on the same segments) under a
*predeclared* cost grid, and classifies cost robustness per asset.

Predeclared cost grid (fixed a-priori, NOT tuned to OOS results)
----------------------------------------------------------------
The repository's "realistic cost model" is the one used by
`examples/ma_crossover_real_data.py`:
    commission_per_trade = 2.0   ($2 per fill)
    commission_per_share = 0.003 ($0.003 per share)
    slippage_cents       = 2.0   (2 cents fixed per share)
    slippage_proportional= 0.0005 (5 bps proportional)
All four components are scaled together by a single multiplier so the grid is
one-dimensional and interpretable:

    gross        (0.0x) : zero cost -- reference; must reproduce the admitted
                          zero-cost momentum evidence.
    realistic_1x (1.0x) : the repository's realistic cost model.
    stress_2x    (2.0x) : a conservative stress (double every component).

The coin-flip null is run through the *same* `cfg` at every level, so the null
is cost-matched (it pays the same costs per unit turnover). The null's seed is
fixed per lookback, so the null draw is identical across cost levels and the
candidate-vs-null comparison isolates the cost effect.

A-priori cost-robustness classification (per asset, across lookbacks {3,5,10})
------------------------------------------------------------------------------
`edge(level, lb)` is True when the regime-stability verdict at that level is
`REGIME_STABLE` AND every segment median is positive (the same EDGE rule used
by `momentum_lookback_sweep.py`).

    COST_ROBUST     : some lookback is an edge at gross AND realistic_1x AND stress_2x
                      (the edge survives realistic costs and a 2x stress).
    COST_FRAGILE    : some lookback is an edge at gross AND realistic_1x but is
                      lost at stress_2x (survives the realistic model, not a 2x stress).
    COST_ELIMINATED : some lookback is a gross edge that is no longer an edge at
                      realistic_1x (realistic costs alone destroy it).
    NO_EDGE         : no lookback is a gross edge (nothing to stress; consistent
                      with the 4 assets the lookback sweep already found NO_EDGE).
    ANOMALOUS       : an edge appears only under costs (would indicate a defect).

Checked in that priority order (ROBUST > FRAGILE > ELIMINATED > NO_EDGE).

Method (fixed, not tuned): seed 42; dataset yf-ohlcv-universe-2009-to-2026-10-03;
manifest integrity + data preflight + leakage review (AMZN/JPM); all 10 collected
tickers; per (lookback, cost level) the `stress_segments_across_tickers` gate;
cost-robustness classification; internal determinism recomputation.

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

# Predeclared realistic cost model (from examples/ma_crossover_real_data.py).
BASE_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
# Predeclared bounded cost grid: (label, multiplier).
COST_GRID = (
    ("gross", 0.0),
    ("realistic_1x", 1.0),
    ("stress_2x", 2.0),
)
COST_LEVELS = tuple(label for label, _ in COST_GRID)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cost_config(multiplier):
    """BacktestConfig for a cost-grid multiplier (0 = gross/zero cost)."""
    return bt.BacktestConfig(
        initial_capital=1e6,
        warmup_periods=WARM,
        commission_per_trade=BASE_COST["commission_per_trade"] * multiplier,
        commission_per_share=BASE_COST["commission_per_share"] * multiplier,
        slippage_cents=BASE_COST["slippage_cents"] * multiplier,
        slippage_proportional=BASE_COST["slippage_proportional"] * multiplier,
    )


def run_universe_level(all_tickers, cost_multiplier):
    """Run the regime-stability gate for one cost level across all lookbacks.

    Returns {lookback: {ticker: {medians, null_medians, verdict, segments}}}.
    """
    cfg = cost_config(cost_multiplier)
    out = {}
    for lb in LOOKBACKS:
        summary = bt.stress_segments_across_tickers(
            tickers=all_tickers,
            signals_fn=bt.momentum_signals,
            regime_labels_fn=lambda c: bt.volatility_blocks(
                c, n_blocks=N_BLOCKS, window=WINDOW),
            baseline=(("lookback", lb),),
            param_grid=[{"lookback": lb}],
            train_window=TRAIN,
            test_window=TEST,
            warmup=WARM,
            overlap_window=OVERLAP,
            cfg=cfg,
            periods_per_year=252,
            min_segment_bars=MIN_SEGMENT_BARS,
        )
        per_ticker = {}
        for ticker in summary.ticker_order:
            res = summary.assets[ticker]
            per_ticker[ticker] = dict(
                segments=[s.name for s in res.scenarios],
                medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
                null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
                verdict=res.overall_verdict,
                candidate_dispersion=round(res.candidate_dispersion, 3),
                null_dispersion=round(res.null_dispersion, 3),
                n_folds=res.n_folds,
            )
        out[lb] = per_ticker
    return out


def is_edge(cell):
    """EDGE rule: REGIME_STABLE verdict with uniformly positive segment medians."""
    return cell["verdict"] == "REGIME_STABLE" and all(m > 0 for m in cell["medians"])


def classify_asset(per_lookback):
    """A-priori cost-robustness classification for one asset.

    per_lookback: {lookback: {level: cell}}.
    """
    edge_sets = {}
    for level in COST_LEVELS:
        edge_sets[level] = {lb for lb in LOOKBACKS if is_edge(per_lookback[str(lb)][level])}
    gross_e, net_e, stress_e = (
        edge_sets["gross"], edge_sets["realistic_1x"], edge_sets["stress_2x"])
    if gross_e & net_e & stress_e:
        return "COST_ROBUST"
    if (gross_e & net_e) - stress_e:
        return "COST_FRAGILE"
    if gross_e - net_e:
        return "COST_ELIMINATED"
    if not gross_e:
        return "ANOMALOUS" if (net_e or stress_e) else "NO_EDGE"
    return "MIXED"


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

    print("\n=== 3. Leakage review (lookback=5 reference, zero cost) ===")
    for ticker in ["AMZN", "JPM"]:
        bars, _dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, leakage [PASS], "
              f"zero-cost total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only.")

    print("\n=== 4. Momentum cost-sensitivity grid (lookback 3 / 5 / 10) ===")
    print("  Cost grid (multiplier on the realistic model "
          "{comm $2/trade, $0.003/share, 2c fixed, 5bps prop}):")
    for label, mult in COST_GRID:
        c = cost_config(mult)
        print(f"    {label:14s} x{mult:<4}: comm/trade={c.commission_per_trade:.2f} "
              f"comm/share={c.commission_per_share:.4f} "
              f"slip_cents={c.slippage_cents:.2f} "
              f"slip_prop={c.slippage_proportional:.5f}")
    print("  Walk-forward per segment: train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 volatility blocks, "
          "matched coin-flip null\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Load the full universe once (same BarSequence objects for every level).
    all_tickers = {}
    for entry in manifest["entries"]:
        all_tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(all_tickers.keys())

    # results[level][lookback][ticker] = cell
    results = {}
    for label, mult in COST_GRID:
        print(f"  running cost level {label} (x{mult}) ...")
        results[label] = run_universe_level(all_tickers, mult)
    print("")

    # Assemble per-asset profiles and classify.
    per_asset = {}
    counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "COST_ELIMINATED": 0,
              "NO_EDGE": 0, "ANOMALOUS": 0, "MIXED": 0}
    robust_assets = []
    for ticker in ticker_order:
        per_lookback = {}
        for lb in LOOKBACKS:
            per_lookback[str(lb)] = {
                label: results[label][lb][ticker] for label in COST_LEVELS
            }
        cls = classify_asset(per_lookback)
        counts[cls] += 1
        if cls == "COST_ROBUST":
            robust_assets.append(ticker)
        per_asset[ticker] = dict(
            cost_robustness=cls,
            lookback_results=per_lookback,
            edge_lookbacks={
                label: [lb for lb in LOOKBACKS if is_edge(results[label][lb][ticker])]
                for label in COST_LEVELS
            },
        )

    print("=== 5. Cost-robustness classification ===")
    print("  counts: " + ", ".join(f"{k}={counts[k]}" for k in
                                   ["COST_ROBUST", "COST_FRAGILE", "COST_ELIMINATED",
                                    "NO_EDGE", "ANOMALOUS", "MIXED"]))
    print(f"  COST_ROBUST assets: {robust_assets}")
    print("")
    print("  Per-asset edge lookbacks by cost level:")
    for ticker in ticker_order:
        p = per_asset[ticker]
        eb = p["edge_lookbacks"]
        print("    {:6s} {:16s} gross={} 1x={} 2x={}".format(
            ticker, p["cost_robustness"], eb["gross"], eb["realistic_1x"], eb["stress_2x"]))
    print("")

    # Median log-return decay for the assets that had a gross edge (informational).
    print("=== 6. Gross -> net median log-return decay (per lookback) ===")
    print("  (assets with a gross edge at any lookback; cells show median of segment medians)")
    for ticker in ticker_order:
        p = per_asset[ticker]
        if not p["edge_lookbacks"]["gross"]:
            continue
        parts = []
        for lb in LOOKBACKS:
            lr = p["lookback_results"][str(lb)]
            def _med(level):
                ms = lr[level]["medians"]
                return float(np.median(ms)) if ms else float("nan")
            parts.append("lb{}:{:.3f}->{:.3f}->{:.3f}".format(
                lb, _med("gross"), _med("realistic_1x"), _med("stress_2x")))
        print(f"    {ticker:6s} " + "  ".join(parts))
    print("")

    # Turnover / cost-incidence diagnostic (lookback 5, full sample). This
    # documents WHY the cost grid barely moves the medians: a low-turnover
    # signal pays little proportional cost. It is descriptive evidence for the
    # classification, not a signal change.
    print("=== 7. Turnover & cost-incidence diagnostic (lookback 5, full sample) ===")
    turnover_diag = {}
    for ticker in ticker_order:
        bars = all_tickers[ticker]
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=5)
        w = np.array([s.weight for s in sig], dtype=np.float64)
        r0 = bt.run_bars(list(bars), sig, cost_config(0.0))
        r1 = bt.run_bars(list(bars), sig, cost_config(1.0))
        tr0 = float(r0.equity_curve[-1] / 1e6 - 1.0)
        tr1 = float(r1.equity_curve[-1] / 1e6 - 1.0)
        turnover_diag[ticker] = dict(
            n_long=int(np.sum(w > 0)), n_short=int(np.sum(w < 0)),
            n_flat=int(np.sum(w == 0)),
            n_flips=int(np.sum(np.abs(np.diff(w)) > 0)),
            fills_gross=len(r0.trades), fills_net=len(r1.trades),
            total_return_gross=round(tr0, 4), total_return_net=round(tr1, 4),
        )
        print("  {:6s} long={:4d} short={} flat={:2d} flips={} fills={}/{} "
              "gross_ret={:+.3f} net_ret={:+.3f}".format(
                  ticker, turnover_diag[ticker]["n_long"],
                  turnover_diag[ticker]["n_short"], turnover_diag[ticker]["n_flat"],
                  turnover_diag[ticker]["n_flips"], turnover_diag[ticker]["fills_gross"],
                  turnover_diag[ticker]["fills_net"], tr0, tr1))
    print("")

    # Determinism: rerun the target subset at the realistic level and assert
    # identical output (independent recomputation path is in the verifier).
    cfg1 = cost_config(1.0)
    det_ok = True
    for ticker in ["AMZN", "JPM"]:
        bars = all_tickers[ticker]
        for lb in LOOKBACKS:
            summary = bt.stress_segments_across_tickers(
                tickers={ticker: bars},
                signals_fn=bt.momentum_signals,
                regime_labels_fn=lambda c: bt.volatility_blocks(
                    c, n_blocks=N_BLOCKS, window=WINDOW),
                baseline=(("lookback", lb),),
                param_grid=[{"lookback": lb}],
                train_window=TRAIN, test_window=TEST, warmup=WARM,
                overlap_window=OVERLAP, cfg=cfg1, periods_per_year=252,
                min_segment_bars=MIN_SEGMENT_BARS,
            )
            res = summary.assets[ticker]
            r2 = dict(
                medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
                null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
                verdict=res.overall_verdict,
            )
            r1 = results["realistic_1x"][lb][ticker]
            r1_slim = {k: r1[k] for k in ("medians", "null_medians", "verdict")}
            if r1_slim != r2:
                det_ok = False
                print(f"  determinism MISMATCH {ticker} lookback {lb}: {r1_slim} vs {r2}")
    print("determinism (realistic_1x, AMZN/JPM, all lookbacks): r1 == r2: {}".format(det_ok))
    assert det_ok, "non-deterministic output at the realistic cost level"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_grid={label: dict(multiplier=mult, **BASE_COST) for label, mult in COST_GRID},
        cost_levels=list(COST_LEVELS),
        classification_rules=(
            "COST_ROBUST: edge at gross & realistic_1x & stress_2x for some lookback; "
            "COST_FRAGILE: edge at gross & realistic_1x but not stress_2x; "
            "COST_ELIMINATED: gross edge not an edge at realistic_1x; "
            "NO_EDGE: no gross edge; ANOMALOUS: edge only under costs."
        ),
        signal_characterization_note=(
            "momentum_signals uses np.mean(np.log(closes[i-lookback+1:i+1])) -- the "
            "mean log PRICE over the window, not the mean log return. For all "
            "collected large-cap prices (> $1) this mean is positive, so the "
            "realized weight is +1 on every non-neutral bar (turnover_diagnostic "
            "shows n_short=0 and <=1 sign flip per asset). The 'momentum' signal is "
            "therefore effectively constant-long, and the cost grid reflects only "
            "constant-dollar rebalancing cost, not directional turnover. The "
            "cost-robustness classification is valid for the signal as implemented; "
            "it must NOT be read as evidence that a direction-dependent short-horizon "
            "momentum strategy survives costs."
        ),
        per_asset=per_asset,
        turnover_diagnostic=turnover_diag,
        universe=dict(
            cost_robustness_counts=counts,
            cost_robust_assets=robust_assets,
            ticker_order=ticker_order,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum transaction-cost sensitivity: exploratory "
          "simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())