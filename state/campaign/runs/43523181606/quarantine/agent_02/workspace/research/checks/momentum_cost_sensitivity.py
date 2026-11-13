"""Real-data transaction-cost sensitivity of short-horizon momentum.

Objective: decide whether the short-horizon momentum bet survives a conservative
transaction-cost stress on the collected 10-asset universe, and classify the
result as cost-robust or cost-fragile.

Two framework defects were found and repaired before this check could mean
anything (see the module docstring of `research/backtest/engine.py` and
`regime_stability.py`):

1. `momentum_signals` / `mean_reversion_signals` used ``mean(log(close))`` over
   the window (the mean log *price*), which is positive for every price above 1.
   Both signals were therefore constant positions (+1 momentum, -1 reversal)
   and the ``lookback`` parameter had no effect. This check uses the repaired
   trailing-log-return signal.
2. Engine slippage was added as a price *level* shift on both buys and sells,
   so it cancelled inside every round trip and charged no cost at all. This
   check uses the repaired adverse-fill cost model (buy above the close, sell
   below it).

A-priori prediction (declared before the cost grid was evaluated; the turnover
figure used to size it is a mechanical zero-cost property of the repaired
signal, measured before any cost result was inspected):

- P1 (mechanism): the repaired signal is a high-turnover, always-fully-invested
  +/-1 bet. Median notional turnover is ~0.29/bar at lookback 10 and ~0.54/bar
  at lookback 3 (0.41/bar at lookback 5), so the expected cost drag per fold is
  approximately ``bps * turnover * test_window``: ~0.024-0.045 per 84-bar fold at
  10 bps/side and ~0.073-0.136 at 30 bps/side.
- P2 (outcome): a drag of that size should erase most of a single-digit
  per-fold gross edge. Prediction: fewer than half of the assets that show a
  positive REGIME_STABLE gross edge retain one at 30 bps/side, i.e. the modal
  classification is COST_FRAGILE.
- Falsifier for P2: if at least half of the gross-edge assets still show a
  positive REGIME_STABLE edge at 30 bps/side, the prediction is wrong and the
  momentum evidence is cost-robust.

Method (fixed a-priori, not tuned): momentum long/short the trailing lookback-day
log return; hold 1 day; daily rebalance; lookbacks {3, 5, 10}; cost levels
{0, 10, 30} bps per side applied through ``slippage_proportional``; the same
walk-forward windows and 4 volatility blocks as the earlier momentum checks; a
matched coin-flip null that pays the same costs on the same segments.

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
COSTS_BPS = (0.0, 10.0, 30.0)
ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"

A_PRIORI_PREDICTION = (
    "P1: cost drag per fold ~= bps * median_turnover_per_bar * test_window, i.e. "
    "~0.024-0.045 per fold at 10 bps/side and ~0.073-0.136 at 30 bps/side "
    "(turnover 0.29-0.54 notional/bar, measured at zero cost before the grid). "
    "P2: fewer than half of the assets with a positive REGIME_STABLE gross edge "
    "retain one at 30 bps/side, so the modal classification is COST_FRAGILE. "
    "Falsifier: >= half of gross-edge assets still show a positive REGIME_STABLE "
    "edge at 30 bps/side."
)


def cost_cfg(bps: float) -> bt.BacktestConfig:
    return bt.BacktestConfig(initial_capital=1e6, slippage_proportional=float(bps) / 10000.0)


def segment_result(ticker, bars, lookback, bps):
    """Walk-forward regime-stability result for one ticker/lookback/cost."""
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
        cfg=cost_cfg(bps),
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


def is_positive_edge(sr):
    """Positive REGIME_STABLE edge: the framework verdict is REGIME_STABLE and
    every segment median is positive."""
    return sr["verdict"] == "REGIME_STABLE" and all(m > 0 for m in sr["medians"])


def classify(gross, by_cost):
    """A-priori cost-robustness classification for one asset/lookback.

    COST_ROBUST   : positive REGIME_STABLE gross edge survives 30 bps/side.
    COST_FRAGILE  : gross edge exists but is lost by 30 bps/side.
    NO_GROSS_EDGE : no positive REGIME_STABLE gross edge to begin with.
    """
    if not is_positive_edge(gross):
        return "NO_GROSS_EDGE"
    if is_positive_edge(by_cost[30.0]):
        return "COST_ROBUST"
    return "COST_FRAGILE"


def cost_mechanics_gate():
    """Charge-mechanics gate: costs must be adverse and monotone.

    (a) a flat-price round trip loses exactly 2 * s * notional;
    (b) terminal equity is strictly decreasing in the slippage level;
    (c) the old price-level-shift convention would leave the round trip at
        exactly initial capital, which is what makes the repair necessary.
    """
    bars = [
        bt.Bar(date=1, open=100.0, high=100.0, low=100.0, close=100.0, volume=1e6),
        bt.Bar(date=2, open=100.0, high=100.0, low=100.0, close=100.0, volume=1e6),
    ]
    signals = [bt.Signal(date=1, weight=1.0), bt.Signal(date=2, weight=0.0)]
    s = 0.001
    res = bt.run_bars(bars, signals, cost_cfg(10.0))
    round_trip = float(res.equity_curve[-1])
    expected = 1e6 - 2.0 * s * 1e6
    gate_a = abs(round_trip - expected) < 1e-6

    np.random.seed(13)
    rb = bt.generate_bars(400, seed=13)
    sigs = [bt.Signal(date=i + 1, weight=1.0 if i % 3 else -1.0) for i in range(len(rb))]
    terminals = []
    for bps in COSTS_BPS:
        r = bt.run_bars(list(rb), sigs, cost_cfg(bps))
        terminals.append(float(r.equity_curve[-1]))
    gate_b = all(hi < lo for lo, hi in zip(terminals, terminals[1:]))
    gate_c = abs(round_trip - 1e6) > 1.0
    return dict(
        round_trip_terminal=round_trip,
        round_trip_expected=expected,
        terminals_by_cost_bps=dict(zip([str(c) for c in COSTS_BPS], terminals)),
        gate_round_trip_exact=gate_a,
        gate_monotone_cost=gate_b,
        gate_cost_is_not_free=gate_c,
        passed=bool(gate_a and gate_b and gate_c),
    )


def terminal_wealth(tickers, lookbacks=LOOKBACKS, costs=COSTS_BPS):
    """Full-sample terminal equity per ticker/lookback/cost.

    The framework's per-fold metric is a *relative* return over the test
    window, so a cost paid before that window lowers the denominator and can
    raise the ratio. Terminal wealth is the base-free cost gauge and is
    reported alongside the framework medians.
    """
    out = {}
    for t, bars in tickers.items():
        closes = bars.closes_array()
        for lb in lookbacks:
            sig = bt.momentum_signals(closes, lookback=lb)
            for bps in costs:
                res = bt.run_bars(list(bars), sig, cost_cfg(bps))
                out.setdefault(t, {}).setdefault(str(lb), {})[str(bps)] = round(
                    float(res.equity_curve[-1]), 2)
    return out


def turnover_profile(tickers):
    out = {}
    for lb in LOOKBACKS:
        turns = []
        for t, bars in tickers.items():
            closes = bars.closes_array()
            sig = bt.momentum_signals(closes, lookback=lb)
            res = bt.run_bars(list(bars), sig, bt.BacktestConfig(initial_capital=1e6))
            gross = sum(abs(f.shares) * f.price for f in res.trades) / 1e6
            turns.append(gross / len(bars))
        out[str(lb)] = dict(
            median=round(float(np.median(turns)), 4),
            min=round(float(np.min(turns)), 4),
            max=round(float(np.max(turns)), 4),
        )
    return out


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

    print("\n=== 3. Leakage review (AMZN, JPM) ===")
    tickers = {}
    for ticker in ("AMZN", "JPM"):
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        sig = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(sig, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), sig, cost_cfg(0.0))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        w = [s.weight for s in sig]
        print(f"  {ticker}: {bars.n_bars} bars, long {sum(1 for x in w if x > 0)} / "
              f"short {sum(1 for x in w if x < 0)} / flat {sum(1 for x in w if x == 0)}, "
              f"leakage [PASS]")

    print("\n=== 4. Charge-mechanics gate (cost model must be adverse and monotone) ===")
    mech = cost_mechanics_gate()
    print(f"  round trip terminal {mech['round_trip_terminal']:.2f} vs expected "
          f"{mech['round_trip_expected']:.2f} -> "
          f"{'OK' if mech['gate_round_trip_exact'] else 'FAIL'}")
    print(f"  terminals by cost {[round(v, 2) for v in mech['terminals_by_cost_bps'].values()]} "
          f"-> {'monotone' if mech['gate_monotone_cost'] else 'NOT MONOTONE'}")
    print(f"  cost is not free: {mech['gate_cost_is_not_free']}")
    if not mech["passed"]:
        print("CHARGE-MECHANICS GATE FAILED; aborting cost grid.")
        sys.exit(1)

    print("\n=== 5. Repaired-signal turnover profile (zero cost, mechanical) ===")
    turn = turnover_profile(tickers)
    print(f"  {turn}")
    wealth = terminal_wealth(tickers)
    for t in ("AMZN", "JPM"):
        print("  terminal wealth {}: {}".format(
            t, {lb: v for lb, v in wealth[t].items()}))
    print("  A-priori prediction: " + A_PRIORI_PREDICTION + "\n")

    print("=== 6. Cost grid (lookbacks 3/5/10 x cost 0/10/30 bps per side) ===")
    print("  Params: walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 volatility blocks, matched "
          "coin-flip null paying the same costs\n".format(
              TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    grid = {}
    for ticker in [e["ticker"] for e in manifest["entries"]]:
        if ticker not in tickers:
            tickers[ticker] = bt.load_ticker(ticker)[0]
        for lb in LOOKBACKS:
            for bps in COSTS_BPS:
                grid[(ticker, lb, bps)] = segment_result(
                    ticker, tickers[ticker], lb, bps)

    per_asset = {}
    class_counts = {"COST_ROBUST": 0, "COST_FRAGILE": 0, "NO_GROSS_EDGE": 0}
    retained = {str(bps): 0 for bps in COSTS_BPS}
    gross_edge_assets = []
    for ticker in [e["ticker"] for e in manifest["entries"]]:
        per_lb = {}
        for lb in LOOKBACKS:
            gross = grid[(ticker, lb, 0.0)]
            by_cost = {bps: grid[(ticker, lb, bps)] for bps in COSTS_BPS}
            cls = classify(gross, by_cost)
            class_counts[cls] += 1
            drag = {str(bps): round(
                float(np.median(gross["medians"])) - float(np.median(by_cost[bps]["medians"])), 3)
                for bps in COSTS_BPS}
            survival = None
            if is_positive_edge(gross):
                ok = [bps for bps in COSTS_BPS if is_positive_edge(by_cost[bps])]
                survival = max(ok) if ok else None
                gross_edge_assets.append((ticker, lb))
            for bps in COSTS_BPS:
                if is_positive_edge(by_cost[bps]):
                    retained[str(bps)] += 1
            per_lb[str(lb)] = dict(
                cost_bps={str(bps): by_cost[bps] for bps in COSTS_BPS},
                cost_drag_vs_gross_median=drag,
                classification=cls,
                survival_cost_bps=survival,
            )
        per_asset[ticker] = per_lb

    print("  Per-asset classification (lookback 5):")
    for ticker, lbv in per_asset.items():
        lb5 = lbv["5"]
        g = lb5["cost_bps"]["0.0"]
        n10 = lb5["cost_bps"]["10.0"]
        n30 = lb5["cost_bps"]["30.0"]
        print("    {:6s} {:14s} gross {} {} | 10bps {} {} | 30bps {} {} | survival {}".format(
            ticker, lb5["classification"],
            g["verdict"], g["medians"],
            n10["verdict"], n10["medians"],
            n30["verdict"], n30["medians"],
            lb5["survival_cost_bps"]))
    print("")

    total_cells = len(per_asset) * len(LOOKBACKS)
    gross_edge_cells = len(gross_edge_assets)
    retained30 = retained["30.0"]
    p2_supported = gross_edge_cells > 0 and retained30 < gross_edge_cells / 2.0
    print("  classification counts across {} asset x lookback cells: {}".format(
        total_cells, class_counts))
    print("  cells with a positive gross edge: {}".format(gross_edge_cells))
    print("  positive-edge cells retained by cost level: {}".format(retained))
    print("  a-priori P2 supported (modal COST_FRAGILE / <half retained at 30bps): {}".format(
        p2_supported))
    print("")

    # Determinism: rerun the AMZN/JPM subset and assert identical output.
    det = {}
    for ticker in ("AMZN", "JPM"):
        for lb in LOOKBACKS:
            for bps in COSTS_BPS:
                det[(ticker, lb, bps)] = segment_result(ticker, tickers[ticker], lb, bps)
    same = all(det[k] == grid[k] for k in det)
    print("determinism: r1 == r2: {}".format(same))
    assert same, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        costs_bps=list(COSTS_BPS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        a_priori_prediction=A_PRIORI_PREDICTION,
        cost_model="adverse fill: buy at close*(1+s)+cents, sell at close*(1-s)-cents",
        null="matched coin-flip null paying the same per-side cost on the same segments",
        charge_mechanics=mech,
        turnover=turn,
        terminal_wealth=wealth,
        per_asset=per_asset,
        universe=dict(
            classification_counts=class_counts,
            gross_edge_cells=gross_edge_cells,
            gross_edge_assets=[list(x) for x in gross_edge_assets],
            positive_edge_cells_by_cost=retained,
            a_priori_p2_supported=p2_supported,
            ticker_order=[e["ticker"] for e in manifest["entries"]],
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("artifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())