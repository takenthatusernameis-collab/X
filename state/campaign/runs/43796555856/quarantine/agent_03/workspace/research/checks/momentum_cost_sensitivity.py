"""Transaction-cost sensitivity of the short-horizon momentum candidate.

Background: activation 37318950814 admitted short-horizon momentum (long the
previous 5-day return, hold 1 day, daily rebalance) as the campaign's first
candidate positive evidence (REGIME_STABLE positive medians in 7/10 collected
assets), and activation 37361000967 showed the pattern is not a single-point
lookback artifact (lookbacks 3/5/10). Every one of those runs used the default
zero-cost `BacktestConfig`. The momentum construction rebalances a
constant-dollar position at *every* bar, so its implied turnover is close to
two times notional per day; a zero-cost verdict on such a construction is not
decision-relevant until it is priced.

A-priori design (fixed BEFORE execution, not tuned after seeing results):

* Cost grid (three levels, ordered):
  1. `zero`        -- all cost fields zero (reproduces the prior evidence).
  2. `realistic`   -- the repository's existing realistic cost model, copied
     verbatim from `examples/ma_crossover_real_data.py`:
     commission_per_trade=2.0, commission_per_share=0.003,
     slippage_cents=2.0, slippage_proportional=0.0005.
  3. `stress_2x`   -- every cost field of `realistic` doubled
     (commission_per_trade=4.0, commission_per_share=0.006,
     slippage_cents=4.0, slippage_proportional=0.0010), a deliberately
     conservative stress, not a calibrated forecast of real execution costs.
* Lookbacks: 3, 5, 10 (the same grid as `momentum_lookback_sweep.py`).
* Universe: the existing 10-asset collected universe, unchanged.
* Gate: the existing walk-forward regime-stability gate (4 contiguous
  volatility blocks, train=252d/test=84d/warmup=60d/overlap=60d, min 400
  bars/segment) with a matched coin-flip null at the SAME cost level, so the
  null absorbs the same cost drag as the candidate.
* Per-cell verdict mapping (identical to `momentum_lookback_sweep.py`):
  REGIME_STABLE with uniformly positive medians -> EDGE;
  REGIME_STABLE_LOSS -> LOSS; anything else -> NO_EDGE.
* Predeclared per-asset classification, in evaluation order:
  - `NO_EDGE_ZERO_COST` : no lookback shows a positive edge at zero cost.
  - `COST_ROBUST`       : >= 2 of 3 lookbacks show a positive edge at the
    `realistic` cost level.
  - `COST_FRAGILE`      : exactly 1 of 3 lookbacks shows a positive edge at the
    `realistic` level.
  - `COST_ELIMINATED`   : no lookback shows a positive edge at `realistic`
    while >= 1 lookback showed one at zero cost.

Section 7 (signal integrity) is a validity audit of the signal under test, not
an additional experiment. It exists because section 3's turnover accounting
(annual turnover ~3.8x capital for a rule documented as rebalancing a
constant-dollar +/-1 position every bar) is inconsistent with the documented
construction, and a cost classification is only interpretable if the signal is
what the record says it is.

Section 8 (corrected-signal diagnostic) is predeclared here, before any result
was observed, because section 7 shows the recorded implementation cannot answer
the primary question. It runs the identical 3x3 grid on a docstring-faithful
log-return momentum (sign of the past `lookback`-day log return), with the same
predeclared classification rule. A-priori prediction for that diagnostic,
recorded before execution: a genuine N-day -> 1-day momentum rule rebalances a
constant-dollar +/-1 position every bar (~2 notional per day, several hundred
times capital per year); at the `realistic` cost level (~5 bps proportional
plus per-share and per-trade charges per side) the round-trip cost is on the
order of tens of percent per year, which should exceed the gross edge of a
short-horizon signal on large caps. The prediction is that most (asset,
lookback) cells that show a zero-cost edge lose it at `realistic`, i.e. most
assets classify COST_ELIMINATED or COST_FRAGILE rather than COST_ROBUST. The
prediction is falsified if >= 2 lookbacks still show a positive REGIME_STABLE
edge at the `realistic` level for most assets.

The classification is descriptive of cost robustness only. It makes no claim
about live tradability: the cost levels are assumptions, the universe is ten
large caps, and no market-impact, borrow, or capacity model is present.

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

COST_LEVELS = (
    ("zero", dict()),
    ("realistic", dict(
        commission_per_trade=2.0,
        commission_per_share=0.003,
        slippage_cents=2.0,
        slippage_proportional=0.0005,
    )),
    ("stress_2x", dict(
        commission_per_trade=4.0,
        commission_per_share=0.006,
        slippage_cents=4.0,
        slippage_proportional=0.0010,
    )),
)
LEVEL_NAMES = tuple(name for name, _ in COST_LEVELS)

CLASSIFICATION_RULE = (
    "NO_EDGE_ZERO_COST if no lookback has a positive REGIME_STABLE edge at zero "
    "cost; else COST_ROBUST if >=2 lookbacks have such an edge at the realistic "
    "level; else COST_FRAGILE if exactly 1 does; else COST_ELIMINATED."
)

CORRECTED_SIGNAL_PREDICTION = (
    "Predeclared before execution: a docstring-faithful N-day -> 1-day momentum "
    "rule rebalances a constant-dollar +/-1 position every bar (about 2x "
    "notional per day, several hundred times capital per year). At the "
    "`realistic` cost level the round-trip cost is on the order of tens of "
    "percent per year and should exceed a short-horizon gross edge, so most "
    "(asset, lookback) cells with a zero-cost edge are predicted to lose it at "
    "`realistic` (COST_ELIMINATED / COST_FRAGILE rather than COST_ROBUST). "
    "Falsified if >=2 lookbacks still show a positive REGIME_STABLE edge at the "
    "`realistic` level for most assets."
)

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
LOOKBACK_SWEEP_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

COST_MODEL_CAVEAT = (
    "BacktestConfig's per-share charges (commission_per_share, slippage_cents) "
    "are not invariant to split adjustment: the collected series are "
    "split-adjusted, so a per-share charge is a different fraction of notional "
    "at different price levels. Only the proportional component "
    "(slippage_proportional, commission_per_trade relative to capital) is "
    "scale-free. Cost levels here are therefore stated assumptions, not "
    "calibrated execution forecasts, and the stress_2x level is a sensitivity "
    "device rather than a realistic estimate."
)


def log_return_momentum_signals(closes, lookback: int = 5):
    """Docstring-faithful momentum: sign of the past ``lookback``-day log return.

    Local diagnostic implementation used only by section 8; the framework
    function `bt.momentum_signals` is left untouched.
    """
    n = len(closes)
    out = [bt.Signal(date=i + 1, weight=0.0) for i in range(n)]
    lookback = int(lookback)
    logs = np.log(np.asarray(closes, dtype=np.float64))
    for i in range(lookback, n):
        ret = float(logs[i] - logs[i - lookback])
        out[i] = bt.Signal(date=i + 1, weight=1.0 if ret > 0 else -1.0)
    return out


def cost_cfg(level: str) -> bt.BacktestConfig:
    """BacktestConfig for one predeclared cost level (constant-dollar sizing)."""
    params = dict(COST_LEVELS)[level]
    return bt.BacktestConfig(initial_capital=1e6, target_exposure=1.0, **params)


def cell_result(ticker_bars, lookback: int, level: str, signals_fn=None):
    """One (asset, lookback, cost level) regime-stability cell."""
    closes = ticker_bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    signals_fn = signals_fn or bt.momentum_signals
    sig = signals_fn(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError("signals length mismatch for {}".format(lookback))
    res = bt.stress_segments(
        signals_fn=signals_fn,
        bars=ticker_bars,
        signals=sig,
        param_grid=[{"lookback": lookback}],
        baseline=(("lookback", lookback),),
        segment_fn=seg_fn,
        train_window=TRAIN,
        test_window=TEST,
        warmup=WARM,
        overlap_window=OVERLAP,
        cfg=cost_cfg(level),
        periods_per_year=252,
        min_segment_bars=MIN_SEGMENT_BARS,
    )
    return dict(
        segments=[s.name for s in res.scenarios],
        medians=[float(s.baseline_median_log_return) for s in res.scenarios],
        null_medians=[float(s.noise_median_log_return) for s in res.scenarios],
        candidate_dispersion=float(res.candidate_dispersion),
        null_dispersion=float(res.null_dispersion),
        n_folds=int(res.n_folds),
        n_folds_per_segment=[int(s.n_folds) for s in res.scenarios],
        verdict=res.overall_verdict,
    )


def cell_verdict(cell) -> str:
    """EDGE / LOSS / NO_EDGE for one cell, using the published rule."""
    if cell["verdict"] == "REGIME_STABLE" and all(m > 0 for m in cell["medians"]):
        return "EDGE"
    if cell["verdict"] == "REGIME_STABLE_LOSS":
        return "LOSS"
    return "NO_EDGE"


def scalar_median(cell) -> float:
    """Median of the cell's per-segment medians (unrounded)."""
    return float(np.median(np.asarray(cell["medians"], dtype=np.float64)))


def classify(n_edge_by_level) -> str:
    """Predeclared per-asset cost-robustness classification."""
    n0 = n_edge_by_level["zero"]
    n1 = n_edge_by_level["realistic"]
    if n0 == 0:
        return "NO_EDGE_ZERO_COST"
    if n1 >= 2:
        return "COST_ROBUST"
    if n1 == 1:
        return "COST_FRAGILE"
    return "COST_ELIMINATED"


def summarise(per_cost, ticker_order):
    """Per-asset profiles, classifications and universe counts for one grid."""
    per_asset = {}
    counts = {c: 0 for c in ("NO_EDGE_ZERO_COST", "COST_ROBUST",
                             "COST_FRAGILE", "COST_ELIMINATED")}
    cells = {level: [] for level in LEVEL_NAMES}
    for ticker in ticker_order:
        profiles = {}
        n_edge = {}
        for level in LEVEL_NAMES:
            n_edge[level] = sum(
                1 for lb in LOOKBACKS
                if cell_verdict(per_cost[level][ticker][str(lb)]) == "EDGE")
            for lb in LOOKBACKS:
                cells[level].append(
                    (ticker, lb, scalar_median(per_cost[level][ticker][str(lb)])))
        for lb in LOOKBACKS:
            profiles[str(lb)] = {
                level: cell_verdict(per_cost[level][ticker][str(lb)])
                for level in LEVEL_NAMES
            }
        base = {str(lb): scalar_median(per_cost["zero"][ticker][str(lb)])
                for lb in LOOKBACKS}
        drag = {}
        for level in LEVEL_NAMES:
            drag[level] = round(float(np.median([
                base[str(lb)] - scalar_median(per_cost[level][ticker][str(lb)])
                for lb in LOOKBACKS])), 4)
        cls = classify(n_edge)
        counts[cls] += 1
        per_asset[ticker] = dict(
            cost_profiles=profiles,
            n_edge_cells=n_edge,
            classification=cls,
            median_cost_drag=drag,
        )
    universe = dict(classification_counts=counts, ticker_order=list(ticker_order),
                    per_level=dict())
    for level in LEVEL_NAMES:
        vals = np.asarray([v for _, _, v in cells[level]], dtype=np.float64)
        n_edge_cells = sum(1 for t in ticker_order for lb in LOOKBACKS
                           if cell_verdict(per_cost[level][t][str(lb)]) == "EDGE")
        n_loss = sum(1 for t in ticker_order for lb in LOOKBACKS
                     if cell_verdict(per_cost[level][t][str(lb)]) == "LOSS")
        universe["per_level"][level] = dict(
            n_edge_cells=n_edge_cells,
            n_loss_cells=n_loss,
            n_cells=len(ticker_order) * len(LOOKBACKS),
            median_cell_median=round(float(np.median(vals)), 4),
            positive_cell_share=round(float(np.mean(vals > 0)), 3),
        )
    return per_asset, universe


def run_grid(tickers, ticker_order, signals_fn=None):
    """Run the full (cost level x lookback x universe) grid."""
    per_cost = {}
    for level in LEVEL_NAMES:
        per_asset = {}
        for ticker in ticker_order:
            per_asset[ticker] = {
                str(lb): cell_result(tickers[ticker], lb, level, signals_fn)
                for lb in LOOKBACKS
            }
        per_cost[level] = per_asset
    return per_cost


def print_grid(per_asset, universe, title):
    print("\n  {}".format(title))
    for ticker in universe["ticker_order"]:
        p = per_asset[ticker]
        cells = " | ".join(
            "{}:{}".format(lb, "/".join(p["cost_profiles"][str(lb)][lv][:4]
                                        for lv in LEVEL_NAMES))
            for lb in LOOKBACKS
        )
        print("    {:6s} {:18s} n_edge={} drag(real)={:+.3f}  {}".format(
            ticker, p["classification"], p["n_edge_cells"],
            p["median_cost_drag"]["realistic"], cells))
    for level in LEVEL_NAMES:
        u = universe["per_level"][level]
        print("    {:10s} EDGE {}/{} | LOSS {} | median of cell medians {:+.4f} "
              "| positive share {:.1f}".format(
                  level, u["n_edge_cells"], u["n_cells"], u["n_loss_cells"],
                  u["median_cell_median"], u["positive_cell_share"]))


def leakage_review(target=("AMZN", "JPM"), signals_fn=None, lookback=5):
    """Leakage gate at the realistic cost level, with turnover accounting."""
    out = {}
    cfg = cost_cfg("realistic")
    signals_fn = signals_fn or bt.momentum_signals
    for ticker in target:
        bars, _dates = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = signals_fn(closes, lookback=lookback)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, cfg)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        traded = sum(abs(f.shares * f.price) for f in res.trades)
        commission = sum(f.commission for f in res.trades)
        years = len(closes) / 252.0
        out[ticker] = dict(
            n_bars=len(closes),
            n_fills=len(res.trades),
            traded_notional_over_capital=round(traded / 1e6, 3),
            annual_turnover_over_capital=round(traded / 1e6 / years, 3),
            commission_over_capital=round(commission / 1e6, 4),
            total_return=round(float(res.equity_curve[-1] / 1e6 - 1.0), 4),
        )
    return out


def signal_integrity(tickers, ticker_order):
    """Validity audit of the framework momentum signal.

    `bt.momentum_signals` computes `np.mean(np.log(closes[i-lookback+1:i+1]))`
    and names it `ret`. That expression is the mean of log PRICES over the
    window, not a return; for any price above 1 it is strictly positive, so the
    weight is always +1.0 after the lookback warmup. This audit records the
    weight set, the minimum window mean log price, and equality with the
    always-long signal so the diagnosis is reproducible rather than asserted.
    """
    per_ticker = {}
    min_mean_log_price = float("inf")
    for ticker in ticker_order:
        closes = tickers[ticker].closes_array()
        entry = {}
        for lb in LOOKBACKS:
            sig = bt.momentum_signals(closes, lookback=lb)
            weights = [s.weight for s in sig]
            after = weights[lb:]
            logs = np.log(np.asarray(closes, dtype=np.float64))
            for i in range(lb, len(closes)):
                min_mean_log_price = min(
                    min_mean_log_price,
                    float(np.mean(logs[i - lb + 1: i + 1])))
            al = bt.always_long_signal(closes, 1, 1)
            entry[str(lb)] = dict(
                n_bars=len(weights),
                n_long_after_warmup=sum(1 for w in after if w > 0),
                n_short_after_warmup=sum(1 for w in after if w < 0),
                n_neutral_after_warmup=sum(1 for w in after if w == 0),
                distinct_weights_after_warmup=sorted(set(after)),
                matches_always_long_after_warmup=(
                    after == [s.weight for s in al][lb:]),
            )
        per_ticker[ticker] = entry
    return dict(
        expression="np.mean(np.log(closes[i - lookback + 1 : i + 1]))",
        intended_expression="np.log(closes[i]) - np.log(closes[i - lookback])",
        min_window_mean_log_price=round(min_mean_log_price, 6),
        per_ticker=per_ticker,
        diagnosis=(
            "The recorded expression is the mean of log PRICES over the window, "
            "not a return. The resulting weight is therefore a function of the "
            "price LEVEL, not of the lookback: it is +1.0 for every bar whose "
            "window geometric-mean adjusted price exceeds 1.0 and -1.0 below it. "
            "Nine of ten collected assets never cross that level and are "
            "constant +1.0 after the warmup; NVDA (split-adjusted early prices "
            "below 1.0) flips sign once when its adjusted price crosses 1.0. "
            "The lookback parameter only delays the start of the position."
        ),
    )


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        print("  {}: {}".format(
            entry["ticker"], "OK" if actual == entry["checksum_sha256"] else "MISMATCH"))
    print("  dataset: {} | collection: {} | universe: {}".format(
        manifest["dataset_id"], manifest["collection_date"], len(manifest["universe"])))

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    print("\n=== 3. Leakage review at the realistic cost level (lookback=5) ===")
    leak = leakage_review()
    for ticker, r in leak.items():
        print("  {}: fills={} traded={}x capital turnover={}x/yr "
              "commission={}x total_return={:+.3f} [leakage PASS]".format(
                  ticker, r["n_fills"], r["traded_notional_over_capital"],
                  r["annual_turnover_over_capital"], r["commission_over_capital"],
                  r["total_return"]))
    print("  NOTE: turnover of a rule documented as rebalancing a constant-dollar "
          "+/-1 position every bar should be on the order of 100x capital per "
          "year; see section 7.")

    print("\n=== 4. Predeclared cost grid x lookback sweep (recorded implementation) ===")
    print("  cost levels: {}".format(LEVEL_NAMES))
    print("  params: train={}d / test={}d / warmup={}d / overlap={}d, "
          "{} volatility blocks, matched coin-flip null per cost level\n".format(
              TRAIN, TEST, WARM, OVERLAP, N_BLOCKS))

    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(tickers.keys())

    per_cost = run_grid(tickers, ticker_order)
    per_asset, universe = summarise(per_cost, ticker_order)
    print_grid(per_asset, universe, "Per-asset cost classification (rule: {})".format(
        CLASSIFICATION_RULE))

    print("\n=== 5. Cross-consistency gate: zero-cost column vs lookback sweep ===")
    cross_ok = True
    if not LOOKBACK_SWEEP_ARTIFACT.exists():
        print("  SKIPPED: {} not present".format(LOOKBACK_SWEEP_ARTIFACT))
        cross_ok = False
    else:
        with open(LOOKBACK_SWEEP_ARTIFACT) as f:
            prior = json.load(f)
        prior_lp = prior["universe"]["per_asset"]
        for ticker in ticker_order:
            pub_med = prior_lp[ticker]["lookback_medians"]
            pub_null = prior_lp[ticker]["lookback_null_medians"]
            for lb in LOOKBACKS:
                cur = per_cost["zero"][ticker][str(lb)]
                pub = prior["per_asset"][ticker][str(lb)]
                ok = (
                    [round(m, 3) for m in cur["medians"]] == pub_med[str(lb)]
                    and [round(m, 3) for m in cur["null_medians"]] == pub_null[str(lb)]
                    and cur["segments"] == pub["segments"]
                    and cur["verdict"] == pub["verdict"]
                )
                if not ok:
                    cross_ok = False
                    print("  MISMATCH {} lookback {}: recomputed {} vs published {}".format(
                        ticker, lb, cur["verdict"], pub["verdict"]))
        print("  cross-consistency vs momentum_lookback_sweep_results.json: {}".format(
            "MATCH" if cross_ok else "MISMATCH"))
        print("  zero-cost edge-cell counts vs published robustness verdicts "
              "(ROBUST means >=2 lookbacks with a positive edge):")
        for ticker in ticker_order:
            print("    {:6s} zero-cost edge cells={} published={}".format(
                ticker, per_asset[ticker]["n_edge_cells"]["zero"],
                prior_lp[ticker]["robustness_verdict"]))

    print("\n=== 6. Determinism (AMZN/JPM subset rerun at every cost level) ===")
    det_ok = True
    for level in LEVEL_NAMES:
        for ticker in ("AMZN", "JPM"):
            again = {str(lb): cell_result(tickers[ticker], lb, level) for lb in LOOKBACKS}
            same = json.dumps(per_cost[level][ticker], sort_keys=True) == \
                json.dumps(again, sort_keys=True)
            det_ok = det_ok and same
            print("  {:10s} {:5s} r1 == r2: {}".format(level, ticker, same))
    assert det_ok, "non-deterministic output"

    print("\n=== 7. Signal integrity audit (validity of the signal under test) ===")
    integrity = signal_integrity(tickers, ticker_order)
    print("  expression under test : {}".format(integrity["expression"]))
    print("  docstring intent      : {}".format(integrity["intended_expression"]))
    print("  min window mean log price over all tickers/lookbacks/bars: {}".format(
        integrity["min_window_mean_log_price"]))
    sample = integrity["per_ticker"]["AMZN"]["5"]
    print("  AMZN lookback 5: distinct weights after warmup={} "
          "long={} short={} neutral={} matches_always_long={}".format(
              sample["distinct_weights_after_warmup"],
              sample["n_long_after_warmup"], sample["n_short_after_warmup"],
              sample["n_neutral_after_warmup"],
              sample["matches_always_long_after_warmup"]))
    print("  DIAGNOSIS: {}".format(integrity["diagnosis"]))
    n_constant_long = sum(
        1 for v in integrity["per_ticker"].values() for lb in LOOKBACKS
        if v[str(lb)]["distinct_weights_after_warmup"] == [1.0]
        and v[str(lb)]["n_short_after_warmup"] == 0)
    flippers = [t for t in ticker_order if any(
        integrity["per_ticker"][t][str(lb)]["n_short_after_warmup"] > 0
        for lb in LOOKBACKS)]
    print("  (asset, lookback) cells constant +1.0 after warmup: {}/{}; "
          "assets with any short bar: {}".format(
              n_constant_long, len(ticker_order) * len(LOOKBACKS), flippers))

    print("\n=== 8. DIAGNOSTIC: corrected log-return momentum through the same grid ===")
    print("  (predeclared before execution; see module docstring)")
    print("  turnover check at the realistic level, corrected signal:")
    corr_leak = leakage_review(signals_fn=log_return_momentum_signals)
    for ticker, r in corr_leak.items():
        print("    {}: fills={} traded={}x capital turnover={}x/yr "
              "commission={}x".format(
                  ticker, r["n_fills"], r["traded_notional_over_capital"],
                  r["annual_turnover_over_capital"], r["commission_over_capital"]))
    corr_cost = run_grid(tickers, ticker_order, log_return_momentum_signals)
    corr_asset, corr_universe = summarise(corr_cost, ticker_order)
    print_grid(corr_asset, corr_universe,
               "Corrected-signal classification (same predeclared rule)")
    corr_det = True
    for level in LEVEL_NAMES:
        for ticker in ("AMZN", "JPM"):
            again = {str(lb): cell_result(tickers[ticker], lb, level,
                                          log_return_momentum_signals)
                     for lb in LOOKBACKS}
            same = json.dumps(corr_cost[level][ticker], sort_keys=True) == \
                json.dumps(again, sort_keys=True)
            corr_det = corr_det and same
    print("  corrected-signal determinism r1 == r2: {}".format(corr_det))
    assert corr_det, "non-deterministic corrected-signal output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        cost_levels=[dict(name=n, **p) for n, p in COST_LEVELS],
        level_names=list(LEVEL_NAMES),
        classification_rule=CLASSIFICATION_RULE,
        cost_model_caveat=COST_MODEL_CAVEAT,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        leakage_recorded_implementation=leak,
        per_cost=per_cost,
        per_asset=per_asset,
        universe=universe,
        cross_consistency_vs_lookback_sweep=bool(cross_ok),
        determinism=bool(det_ok),
        signal_integrity=integrity,
        corrected_signal_diagnostic=dict(
            implementation="research/checks/momentum_cost_sensitivity.py::"
                           "log_return_momentum_signals (local, framework untouched)",
            signals="sign of the past lookback-day log return, hold 1 day, daily rebalance",
            a_priori_prediction=CORRECTED_SIGNAL_PREDICTION,
            leakage=corr_leak,
            per_cost=corr_cost,
            per_asset=corr_asset,
            universe=corr_universe,
            determinism=bool(corr_det),
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Cost levels are stated assumptions, not calibrated "
          "execution forecasts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())