"""Bounded transaction-cost sensitivity check for the qualified momentum edge.

Task R-001 (campaign slot 02). Primary question: does the short-horizon
momentum edge survive conservative transaction-cost stress on the collected
real-data universe?

Scope (fixed a-priori, not tuned to results):
- Use the existing 10-asset collected universe and the existing momentum
  implementation (``research.backtest.regime_stability.momentum_signals``) and
  the existing walk-forward / regime-stability / matched-null machinery.
- Do not change the signal rule, expand the universe, or add leverage.

Predeclared cost grid (three levels; the middle level is the repository's
canonical "realistic" cost model taken verbatim from
``examples/ma_crossover.py`` / ``examples/ma_crossover_real_data.py``):
  zero       : no commission, no slippage (reproduces the published result).
  realistic  : commission_per_trade=2.0, commission_per_share=0.003,
               slippage_cents=2.0, slippage_proportional=0.0005.
  stress2x   : every cost parameter doubled (conservative stress):
               commission_per_trade=4.0, commission_per_share=0.006,
               slippage_cents=4.0, slippage_proportional=0.001.

Predeclared classification rule (evaluated per lookback L in {3,5,10}):
  For each cost level C, run the universe regime-stability gate
  (``stress_segments_across_tickers``) with the corresponding ``BacktestConfig``.
  For each asset: candidate segment medians, matched coin-flip null medians,
  verdict. Define an asset as a "positive edge" at (L,C) if its verdict is
  REGIME_STABLE and every candidate segment median is > 0.
    survives(C) = (n_pos_edge(L,C) >= 1) AND (universe median > 0),
  where the universe median is the median over the 10 assets of each asset's
  median segment median.
    COST_ROBUST      : survives at zero, realistic, and stress2x.
    COST_MARGINAL    : survives at zero and realistic but not at stress2x.
    COST_FRAGILE     : survives at zero but not at realistic.
    NO_EDGE_BASELINE : does not survive at zero (reproduction failure).

Signal-definition diagnostic (recorded, not changed): a genuine daily-rebalanced
momentum strategy flips between long and short and therefore pays material
turnover costs. The check therefore records, per ticker and lookback:
  - the weight distribution and the number of *active-region* sign flips;
  - whether the signal equals a price-level threshold rule
    (weight = +1 iff mean log PRICE over the window > 0);
  - the agreement between the signal and the intended lookback-return sign
    (weight = +1 iff log(closes[i]) - log(closes[i-lookback]) > 0).
If the signal matches the price-level rule but agrees with the intended
lookback-return sign only at chance, the "momentum" edge is not a momentum
signal and its cost robustness cannot validate a momentum edge.

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

# Predeclared cost grid (see module docstring).
COST_LEVELS = {
    "zero": dict(
        commission_per_trade=0.0, commission_per_share=0.0,
        slippage_cents=0.0, slippage_proportional=0.0,
    ),
    "realistic": dict(
        commission_per_trade=2.0, commission_per_share=0.003,
        slippage_cents=2.0, slippage_proportional=0.0005,
    ),
    "stress2x": dict(
        commission_per_trade=4.0, commission_per_share=0.006,
        slippage_cents=4.0, slippage_proportional=0.001,
    ),
}

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"


def cost_cfg(level):
    return bt.BacktestConfig(**COST_LEVELS[level])


def load_universe():
    manifest = load_manifest()
    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    return manifest, tickers


def signal_definition(tickers):
    """Per-ticker, per-lookback weight distribution and rule-identification.

    Identifies what ``momentum_signals`` actually computes by testing it
    against (a) a price-level threshold rule and (b) the intended
    lookback-return sign rule.
    """
    per_ticker = {}
    all_match_price_level = True
    agreements = []
    n_constant_long = 0
    n_tickers = 0
    for ticker, bars in tickers.items():
        closes = bars.closes_array()
        n = len(closes)
        per_lb = {}
        for lb in LOOKBACKS:
            w = np.array([s.weight for s in bt.momentum_signals(closes, lookback=lb)],
                         dtype=np.float64)
            idx = np.arange(lb, n)
            wa = w[lb:]  # active region (after the lookback warmup window)
            n_pos = int(np.sum(wa > 0))
            n_neg = int(np.sum(wa < 0))
            n_zero = int(np.sum(wa == 0))
            active_flips = int(np.sum(np.abs(np.diff(wa)) > 0))

            # (a) price-level threshold rule the code actually implements.
            mean_log_price = np.array(
                [float(np.mean(np.log(closes[i - lb + 1:i + 1]))) for i in idx])
            price_rule = np.where(mean_log_price > 0.0, 1.0, -1.0)
            match_price_level = bool(np.array_equal(wa, price_rule))

            # (b) intended lookback-return sign rule (the docstring's "momentum").
            lb_ret = np.array(
                [float(np.log(closes[i]) - np.log(closes[i - lb])) for i in idx])
            return_rule = np.where(lb_ret > 0.0, 1.0, -1.0)
            agreement = float(np.mean(wa == return_rule))

            # Concrete contradiction: a bar where the level rule says long but
            # the lookback return is negative (the signal goes long after a fall).
            contra = np.where((mean_log_price > 0.0) & (lb_ret < 0.0))[0]

            per_lb[str(lb)] = dict(
                n_pos=n_pos, n_neg=n_neg, n_zero=n_zero,
                active_flips=active_flips,
                constant_long=bool(n_neg == 0 and n_zero == 0),
                match_price_level_rule=match_price_level,
                agreement_with_return_sign=round(agreement, 4),
                mean_log_price_sample=round(float(mean_log_price[n // 2]), 6),
                lookback_return_sample=round(float(lb_ret[n // 2]), 6),
                n_contradiction_bars=int(len(contra)),
            )
            agreements.append(agreement)
            if not match_price_level:
                all_match_price_level = False
        n_tickers += 1
        if all(per_lb[str(lb)]["constant_long"] for lb in LOOKBACKS):
            n_constant_long += 1
        per_ticker[ticker] = per_lb

    agreements = np.array(agreements, dtype=np.float64)
    return dict(
        per_ticker=per_ticker,
        n_tickers=n_tickers,
        n_constant_long=n_constant_long,
        all_match_price_level_rule=all_match_price_level,
        median_agreement_with_return_sign=round(float(np.median(agreements)), 4),
        max_agreement_with_return_sign=round(float(np.max(agreements)), 4),
        min_agreement_with_return_sign=round(float(np.min(agreements)), 4),
    )


def run_cost_cell(tickers, lookback, level):
    """Run the universe regime-stability gate for one (lookback, cost) cell."""
    summary = bt.stress_segments_across_tickers(
        tickers=tickers,
        signals_fn=lambda c, **kw: bt.momentum_signals(c, lookback=lookback),
        regime_labels_fn=lambda c: bt.volatility_blocks(c, n_blocks=N_BLOCKS, window=WINDOW),
        baseline=(("lookback", lookback),),
        param_grid=[{"lookback": lookback}],
        train_window=TRAIN, test_window=TEST, warmup=WARM, overlap_window=OVERLAP,
        cfg=cost_cfg(level), min_segment_bars=MIN_SEGMENT_BARS,
    )
    per_asset = {}
    asset_rep = []
    pos_edge_assets = []
    for ticker in summary.ticker_order:
        r = summary.assets[ticker]
        medians = [round(s.baseline_median_log_return, 3) for s in r.scenarios]
        null_medians = [round(s.noise_median_log_return, 3) for s in r.scenarios]
        rep = float(np.median(medians)) if medians else 0.0
        asset_rep.append(rep)
        pos_edge = (r.overall_verdict == "REGIME_STABLE"
                    and all(m > 0 for m in medians))
        if pos_edge:
            pos_edge_assets.append(ticker)
        per_asset[ticker] = dict(
            segments=[s.name for s in r.scenarios],
            medians=medians,
            null_medians=null_medians,
            candidate_dispersion=round(r.candidate_dispersion, 3),
            null_dispersion=round(r.null_dispersion, 3),
            verdict=r.overall_verdict,
            n_folds=r.n_folds,
            rep_median=round(rep, 3),
            positive_edge=bool(pos_edge),
        )
    universe_median = float(np.median(asset_rep)) if asset_rep else 0.0
    verdict_counts = dict(summary.verdict_counts)
    n_pos_edge = len(pos_edge_assets)
    survives = (n_pos_edge >= 1) and (universe_median > 0.0)
    return dict(
        per_asset=per_asset,
        universe=dict(
            n_pos_edge=n_pos_edge,
            pos_edge_assets=pos_edge_assets,
            universe_median=round(universe_median, 3),
            verdict_counts=verdict_counts,
            survives=bool(survives),
        ),
    )


def classify(lookback_cells):
    """Apply the predeclared per-lookback cost-robustness classification."""
    s = {lvl: lookback_cells[lvl]["universe"]["survives"] for lvl in COST_LEVELS}
    if not s["zero"]:
        cls = "NO_EDGE_BASELINE"
    elif s["zero"] and s["realistic"] and s["stress2x"]:
        cls = "COST_ROBUST"
    elif s["zero"] and s["realistic"] and not s["stress2x"]:
        cls = "COST_MARGINAL"
    elif s["zero"] and not s["realistic"]:
        cls = "COST_FRAGILE"
    else:
        cls = "UNCLASSIFIED"
    return dict(survives=s, mechanical_cost_class=cls)


def turnover_profile(tickers):
    """Quantify cost drag from a full-sample run (lookback 5, realistic cost).

    Counts only active-region sign flips (the neutral->active warmup edge is
    excluded). A genuine daily-rebalanced momentum strategy flips sign often
    and pays large turnover costs; a level-based/static long pays only small
    rebalancing costs.
    """
    cfg = cost_cfg("realistic")
    out = {}
    for ticker, bars in tickers.items():
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        res = bt.run_bars(list(bars), signals, cfg)
        gross_notional = sum(abs(f.shares) * f.price for f in res.trades)
        total_comm = sum(f.commission for f in res.trades)
        n_years = len(closes) / 252.0
        w = np.array([s.weight for s in signals], dtype=np.float64)
        active = w[5:]  # skip the neutral warmup region
        out[ticker] = dict(
            n_trades=len(res.trades),
            gross_turnover_per_year=round(gross_notional / cfg.initial_capital / n_years, 3),
            total_commission_frac=round(total_comm / cfg.initial_capital, 5),
            active_sign_flips=int(np.sum(np.abs(np.diff(active)) > 0)),
        )
    return out


def decide_relevant_class(sigdef):
    """Decision-relevant classification of what the cost test actually measured."""
    if (sigdef["all_match_price_level_rule"]
            and sigdef["median_agreement_with_return_sign"] < 0.70):
        return (
            "CONFOUNDED_PRICE_LEVEL_SIGNAL: momentum_signals implements a "
            "price-level threshold (weight=+1 iff mean log PRICE over the window "
            "> 0), not a lookback-return momentum rule; it agrees with the "
            "intended lookback-return sign only at ~chance. The qualified edge is "
            "a level-dependent/static long, so its negligible cost sensitivity "
            "does not validate a short-horizon momentum edge."
        )
    return "GENUINE_SIGNAL_COST_TESTED"


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]

    print("=== 1. Manifest integrity ===")
    for entry in manifest["entries"]:
        actual = sha256_file(Path(entry["location"]))
        status = "OK" if actual == entry["checksum_sha256"] else "MISMATCH"
        print(f"  {entry['ticker']}: {status}")
    print(f"  dataset: {manifest['dataset_id']} | universe: {len(manifest['universe'])}")

    print("\n=== 2. Data preflight (research/data/preflight.py) ===")
    sys.path.insert(0, str(Path.cwd() / "research" / "data"))
    from importlib import import_module
    pf = import_module("preflight")
    rc = pf.main()
    if rc != 0:
        print("PREFLIGHT FAILED; aborting real-data run.")
        sys.exit(1)

    _, tickers = load_universe()
    ticker_order = list(tickers.keys())

    print("\n=== 3. Leakage review (AMZN/JPM, lookback 5) ===")
    for ticker in ["AMZN", "JPM"]:
        bars = tickers[ticker]
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, bt.BacktestConfig(initial_capital=1e6))
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")

    print("\n=== 4. Signal-definition diagnostic (momentum_signals) ===")
    sigdef = signal_definition(tickers)
    for ticker in ticker_order:
        row = sigdef["per_ticker"][ticker]
        parts = []
        for lb in LOOKBACKS:
            d = row[str(lb)]
            parts.append(
                f"lb{lb}: +1={d['n_pos']}/-1={d['n_neg']} flips={d['active_flips']} "
                f"price_rule={d['match_price_level_rule']} "
                f"agree_ret={d['agreement_with_return_sign']:.2f}")
        print(f"  {ticker}: " + " | ".join(parts))
    print(f"  n_tickers={sigdef['n_tickers']} n_constant_long={sigdef['n_constant_long']} "
          f"all_match_price_level={sigdef['all_match_price_level_rule']} "
          f"median_agree_with_return_sign={sigdef['median_agreement_with_return_sign']}")
    print("  NOTE: agreement_with_return_sign ~ 0.5 means the signal is unrelated to "
          "the sign of the trailing lookback return (i.e. not a momentum rule).")

    print("\n=== 5. Cost-sensitivity grid (lookbacks 3/5/10 x zero/realistic/stress2x) ===")
    grid = {}
    summary = {}
    for lb in LOOKBACKS:
        grid[str(lb)] = {}
        for level in COST_LEVELS:
            cell = run_cost_cell(tickers, lb, level)
            grid[str(lb)][level] = cell
            u = cell["universe"]
            print(f"  lb={lb:2d} {level:9s}: n_pos_edge={u['n_pos_edge']} "
                  f"universe_median={u['universe_median']:+.3f} "
                  f"survives={u['survives']} verdicts={u['verdict_counts']}")
        summary[str(lb)] = classify(grid[str(lb)])
        print(f"  -> lb={lb}: {summary[str(lb)]}")

    print("\n=== 6. Turnover / cost magnitude (lookback 5, realistic) ===")
    turnover = turnover_profile(tickers)
    for ticker in ticker_order:
        t = turnover[ticker]
        print(f"  {ticker}: trades={t['n_trades']} "
              f"gross_turnover/yr={t['gross_turnover_per_year']:.3f} "
              f"comm_frac={t['total_commission_frac']:.5f} "
              f"active_flips={t['active_sign_flips']}")

    print("\n=== 7. Determinism (rerun AMZN/JPM subset) ===")
    sub = {k: tickers[k] for k in ["AMZN", "JPM"]}
    cell_a = run_cost_cell(sub, 5, "realistic")
    cell_b = run_cost_cell(sub, 5, "realistic")
    det = json.dumps(cell_a, sort_keys=True) == json.dumps(cell_b, sort_keys=True)
    print("determinism: r1 == r2: {}".format(det))
    assert det, "non-deterministic output"

    decision_class = decide_relevant_class(sigdef)

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        cost_levels=COST_LEVELS,
        classification_rule=(
            "survives(C) = n_pos_edge>=1 and universe_median>0; "
            "COST_ROBUST/COST_MARGINAL/COST_FRAGILE/NO_EDGE_BASELINE as documented"
        ),
        signal_definition=sigdef,
        cost_robustness_classification=summary,
        cost_grid=grid,
        turnover=turnover,
        decision_relevant_classification=decision_class,
        ticker_order=ticker_order,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("decision_relevant_classification: {}".format(decision_class))
    print("NOTE: research/simulation only. No live trading or production execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())