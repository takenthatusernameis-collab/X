"""Real-data transaction-cost sensitivity of the admitted short-horizon
momentum candidate (campaign slot 02, task R-001).

Question
--------
Does the short-horizon momentum edge survive conservative transaction-cost
stress on the collected real-data universe?

Contract and predeclared design (fixed a-priori, before any cost result is
seen; not tuned afterwards):

* Universe: the existing 10-asset collected universe
  (`yf-ohlcv-universe-2009-to-2026-10-03`).
* Signal: the existing `bt.momentum_signals` implementation, lookbacks 3 / 5 /
  10, hold one day, daily rebalance. No signal, lookback, universe, or sizing
  change is made by this check.
* Gate: the existing walk-forward regime-stability gate
  (`stress_segments`) with the same settings as
  `momentum_lookback_sweep.py`: train=252 / test=84 / warmup=60 / overlap=60,
  4 contiguous volatility blocks (`volatility_blocks`, window=60),
  min 400 bars per segment.
* Matched null: the framework's coin-flip `noise_benchmark`, run on the same
  segments with the same cost configuration at every cost level.
* Cost grid: the enterprise's existing realistic cost vector multiplied by
  {0, 1, 2, 5}. The realistic vector is the one recorded in
  `examples/ma_crossover.py`, `examples/volatility_regime_filter.py`, and
  `logs/ACTIVATION-2026-10-04.md`:
  commission_per_trade=2.0, commission_per_share=0.003, slippage_cents=2.0,
  slippage_proportional=0.0005.
* Classification rule (fixed below, before the grid is evaluated):
  - NO_BASELINE_EDGE: at zero cost the asset is not REGIME_STABLE with a
    uniformly positive median.
  - COST_ERASED: at 1x the median-of-segment-medians is <= 0 or does not
    exceed the matched null's median-of-segment-medians.
  - COST_FRAGILE: survives 1x but at 2x it is <= 0, does not exceed the matched
    null, or retains < 50% of its zero-cost median.
  - COST_ROBUST: survives 1x and 2x with >= 50% retention and exceeds the
    matched null at both levels. The 5x column is reported as a diagnostic
    (`survives_5x`) and is not required for COST_ROBUST.

Diagnostics included because they are decision-relevant, not scope expansion:
the per-lookback signal composition (long/short/neutral shares) and a direct
comparison of the implemented signal definition against the textbook
lookback-return sign. These document what the cost grid is actually measuring.

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
LOOKBACKS = (3, 5, 10)
TRAIN, TEST, WARM, OVERLAP = 252, 84, 60, 60
WINDOW = 60
N_BLOCKS = 4
MIN_SEGMENT_BARS = 400

BASE_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)
COST_GRID = [
    ("zero", {}),
    ("1x", dict(BASE_COST)),
    ("2x", {k: 2.0 * v for k, v in BASE_COST.items()}),
    ("5x", {k: 5.0 * v for k, v in BASE_COST.items()}),
]
LEVEL_NAMES = tuple(name for name, _ in COST_GRID)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cfg_for(level: str) -> bt.BacktestConfig:
    for name, costs in COST_GRID:
        if name == level:
            return bt.BacktestConfig(initial_capital=1e6, **costs)
    raise KeyError(level)


def run_cell(bars, lookback: int, level: str) -> dict:
    """One (asset, lookback, cost-level) walk-forward regime-gate cell."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    signals = bt.momentum_signals(closes, lookback=lookback)
    if len(signals) != len(closes):
        raise ValueError("signals length mismatch")
    res = bt.stress_segments(
        signals_fn=bt.momentum_signals,
        bars=bars,
        signals=signals,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
        cfg=cfg_for(level),
    )
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[round(float(s.baseline_median_log_return), 6) for s in res.scenarios],
        null_medians=[round(float(s.noise_median_log_return), 6) for s in res.scenarios],
        candidate_dispersion=round(float(res.candidate_dispersion), 6),
        null_dispersion=round(float(res.null_dispersion), 6),
        verdict=res.overall_verdict,
        n_folds=int(res.n_folds),
        n_periods=int(res.n_periods),
    )


def composition(closes, lookback: int) -> dict:
    """Signal composition and implemented-vs-textbook sign agreement."""
    w = np.array([s.weight for s in bt.momentum_signals(closes, lookback)], dtype=np.float64)
    n = len(closes)
    implemented = np.sign(
        np.array([float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
                  for i in range(lookback, n)])
    )
    textbook = np.sign(
        np.array([float(np.log(closes[i]) - np.log(closes[i - lookback]))
                  for i in range(lookback, n)])
    )
    both_defined = (implemented != 0) & (textbook != 0)
    disagree = float(np.mean(implemented[both_defined] != textbook[both_defined]))
    return dict(
        long_share=round(float(np.mean(w > 0)), 6),
        short_share=round(float(np.mean(w < 0)), 6),
        neutral_bars=int(np.sum(w == 0)),
        textbook_sign_disagreement=round(disagree, 6),
        degenerate_static_long=bool(float(np.mean(w < 0)) == 0.0
                                    and float(np.mean(w > 0)) >= 0.95),
    )


def round_trip_slippage_probe() -> dict:
    """Two-bar round trip: does the cost vector charge slippage?

    A completed round trip must pay the spread on both legs. In the existing
    engine every fill is priced at ``close + slippage``; this probe records the
    round-trip PnL under each cost component so the artifact states what the
    cost grid actually charges.
    """
    bars = [
        bt.Bar(date=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1e6),
        bt.Bar(date=2, open=110.0, high=110.0, low=110.0, close=110.0, volume=1e6),
    ]
    sig = [bt.Signal(date=1, weight=1.0), bt.Signal(date=2, weight=0.0)]
    out = {}
    for label, costs in (
        ("no_costs", {}),
        ("fixed_2c", dict(slippage_cents=2.0)),
        ("proportional_5bps", dict(slippage_proportional=0.0005)),
        ("realistic_1x", dict(BASE_COST)),
    ):
        res = bt.run_bars(bars, sig, bt.BacktestConfig(initial_capital=100_000.0, **costs))
        out[label] = round(float(res.equity_curve[-1] - 100_000.0), 6)
    return dict(
        round_trip_pnl=out,
        fixed_slippage_cancels=bool(out["fixed_2c"] == out["no_costs"]),
        proportional_slippage_adds=bool(out["proportional_5bps"] > out["no_costs"]),
        realistic_vector_pnl=out["realistic_1x"],
    )


def cost_accounting(ticker: str, bars, lookback: int, level: str = "1x") -> dict:
    """Decompose the cash impact of the cost vector on one full sample."""
    closes = bars.closes_array()
    signals = bt.momentum_signals(closes, lookback=lookback)
    cfg = cfg_for(level)
    res = bt.run_bars(list(bars), signals, cfg)
    trade_closes = np.array([closes[b.date - 1] for b in res.trades], dtype=np.float64)
    shares = np.array([f.shares for f in res.trades], dtype=np.float64)
    commission = float(sum(f.commission for f in res.trades))
    engine_slippage_cash = float(np.sum(-shares * (np.array([f.price for f in res.trades])
                                                   - trade_closes)))
    costs = dict(BASE_COST)
    gross_intended = float(np.sum(
        (costs["slippage_cents"] / 100.0 + trade_closes * costs["slippage_proportional"])
        * np.abs(shares)))
    return dict(
        fills=len(res.trades),
        traded_notional=round(float(np.sum(np.abs(shares) * trade_closes)), 2),
        commission=round(commission, 2),
        engine_slippage_cash_impact=round(engine_slippage_cash, 2),
        gross_intended_slippage_charge=round(gross_intended, 2),
        total_return=round(float((res.equity_curve[-1] - 1e6) / 1e6), 6),
        commission_pct_capital=round(commission / 1e6, 6),
        engine_slippage_pct_capital=round(engine_slippage_cash / 1e6, 6),
        gross_intended_slippage_pct_capital=round(gross_intended / 1e6, 6),
    )


def classify(cells: dict) -> tuple[str, dict]:
    """Apply the predeclared classification rule to one (asset, lookback)."""
    zero = cells["zero"]
    if not (zero["verdict"] == "REGIME_STABLE"
            and all(m > 0 for m in zero["medians"])):
        return "NO_BASELINE_EDGE", {}

    med = {lvl: float(np.median(cells[lvl]["medians"])) for lvl in LEVEL_NAMES}
    null = {lvl: float(np.median(cells[lvl]["null_medians"])) for lvl in LEVEL_NAMES}
    retention = {lvl: (med[lvl] / med["zero"] if med["zero"] > 0 else float("nan"))
                 for lvl in LEVEL_NAMES}

    if med["1x"] <= 0.0 or med["1x"] <= null["1x"]:
        cls = "COST_ERASED"
    elif med["2x"] <= 0.0 or med["2x"] <= null["2x"] or retention["2x"] < 0.5:
        cls = "COST_FRAGILE"
    else:
        cls = "COST_ROBUST"

    detail = dict(
        median_by_level={lvl: round(v, 6) for lvl, v in med.items()},
        null_median_by_level={lvl: round(v, 6) for lvl, v in null.items()},
        retention_by_level={lvl: round(v, 6) for lvl, v in retention.items()},
        exceeds_null_by_level={lvl: bool(med[lvl] > null[lvl]) for lvl in LEVEL_NAMES},
        survives_5x=bool(med["5x"] > 0.0 and med["5x"] > null["5x"]),
    )
    return cls, detail


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
        return 1

    print("\n=== 3. Leakage review under the realistic cost model (AMZN, JPM) ===")
    for ticker in ("AMZN", "JPM"):
        bars, _dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cfg_for("1x"))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, signals {len(signals)}, leakage [PASS], "
              f"costed total_return {total_return:+.3f}")

    tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(tickers.keys())

    print("\n=== 4. Signal composition and definition diagnostic ===")
    print("  (documents what the cost grid measures; no result is tuned from this)")
    composition_by_lookback = {str(lb): {} for lb in LOOKBACKS}
    for ticker in ticker_order:
        closes = tickers[ticker].closes_array()
        for lb in LOOKBACKS:
            composition_by_lookback[str(lb)][ticker] = composition(closes, lb)
        c5 = composition_by_lookback["5"][ticker]
        print("  {:6s} lb5 long={:.4f} short={:.4f} neutral={:2d} "
              "textbook_disagreement={:.4f} degenerate={}".format(
                  ticker, c5["long_share"], c5["short_share"], c5["neutral_bars"],
                  c5["textbook_sign_disagreement"], c5["degenerate_static_long"]))

    print("\n=== 5. Cost grid: lookback x cost level x collected universe ===")
    print("  cost levels: " + ", ".join(
        "{}={}".format(name, costs if costs else "zero") for name, costs in COST_GRID))
    cells = {t: {str(lb): {} for lb in LOOKBACKS} for t in ticker_order}
    for ticker in ticker_order:
        bars = tickers[ticker]
        for lb in LOOKBACKS:
            for level in LEVEL_NAMES:
                cells[ticker][str(lb)][level] = run_cell(bars, lb, level)
        summary = []
        for lb in LOOKBACKS:
            cls, _ = classify(cells[ticker][str(lb)])
            summary.append("lb{}={}".format(lb, cls))
        print("  {:6s} {}".format(ticker, " ".join(summary)))

    print("\n=== 6. Cost-robustness classification (predeclared rule) ===")
    classification = {}
    counts_by_lookback = {str(lb): {"COST_ROBUST": 0, "COST_FRAGILE": 0,
                                    "COST_ERASED": 0, "NO_BASELINE_EDGE": 0}
                          for lb in LOOKBACKS}
    for ticker in ticker_order:
        classification[ticker] = {}
        for lb in LOOKBACKS:
            cls, detail = classify(cells[ticker][str(lb)])
            classification[ticker][str(lb)] = dict(cls=cls, detail=detail)
            counts_by_lookback[str(lb)][cls] += 1
    for lb in LOOKBACKS:
        print("  lookback {}: {}".format(lb, counts_by_lookback[str(lb)]))
    print("  per-asset lookback-5 class:")
    for ticker in ticker_order:
        c = classification[ticker]["5"]
        d = c["detail"]
        if d:
            print("    {:6s} {:16s} med 0/1x/2x/5x = {:.4f}/{:.4f}/{:.4f}/{:.4f} "
                  "retention 1x/2x = {:.4f}/{:.4f} survives_5x={}".format(
                      ticker, c["cls"],
                      d["median_by_level"]["zero"], d["median_by_level"]["1x"],
                      d["median_by_level"]["2x"], d["median_by_level"]["5x"],
                      d["retention_by_level"]["1x"], d["retention_by_level"]["2x"],
                      d["survives_5x"]))
        else:
            print("    {:6s} {}".format(ticker, c["cls"]))

    print("\n=== 7. Cost-accounting audit (what the grid actually charges) ===")
    round_trip = round_trip_slippage_probe()
    print("  round-trip PnL: " + ", ".join(
        "{}={:+.2f}".format(k, v) for k, v in round_trip["round_trip_pnl"].items()))
    print("  fixed slippage cancels: {} | proportional slippage adds: {}".format(
        round_trip["fixed_slippage_cancels"], round_trip["proportional_slippage_adds"]))
    accounting = {t: cost_accounting(t, tickers[t], 5) for t in ticker_order}
    for ticker in ticker_order:
        a = accounting[ticker]
        print("  {:6s} commission={:.2%} | engine slippage cash={:+.2%} | "
              "gross intended slippage={:.2%} | total_return={:+.3f}".format(
                  ticker, a["commission_pct_capital"], a["engine_slippage_pct_capital"],
                  a["gross_intended_slippage_pct_capital"], a["total_return"]))

    print("\n=== 8. Determinism (AMZN lookback 5 at 1x, recomputed) ===")
    bars = tickers["AMZN"]
    again = run_cell(bars, 5, "1x")
    deterministic = again == cells["AMZN"]["5"]["1x"]
    print("  r1 == r2: {}".format(deterministic))
    assert deterministic, "non-deterministic cost-cell output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_grid=[dict(name=name, multiplier=(0.0 if name == "zero" else
                                               {"1x": 1.0, "2x": 2.0, "5x": 5.0}[name]),
                        costs=costs) for name, costs in COST_GRID],
        classification_rule=(
            "NO_BASELINE_EDGE if zero-cost verdict != REGIME_STABLE or any zero-cost "
            "median <= 0; else COST_ERASED if 1x median-of-medians <= 0 or <= 1x matched "
            "null; else COST_FRAGILE if 2x median-of-medians <= 0 or <= 2x matched null "
            "or 2x retention < 0.5; else COST_ROBUST. 5x is a reported diagnostic."
        ),
        signal_composition=composition_by_lookback,
        cost_model_audit=dict(
            round_trip=round_trip,
            per_asset=accounting,
            note=("the existing engine prices every fill at close + slippage, so a "
                  "completed round trip cancels fixed and proportional slippage; only "
                  "commissions are a genuine cost. The gross intended slippage column "
                  "is the conservative charge the cost vector nominally describes."),
        ),
        cells=cells,
        classification=classification,
        counts_by_lookback=counts_by_lookback,
        ticker_order=ticker_order,
        determinism=dict(cell="AMZN lookback 5 at 1x", r1_eq_r2=bool(deterministic)),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Cost sensitivity of the implemented momentum candidate: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())