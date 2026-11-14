"""Bounded transaction-cost sensitivity of the short-horizon momentum edge.

Objective (R-001): test whether the candidate positive momentum evidence
(long the previous N-day return, hold 1 day, daily rebalance; lookback in
{3, 5, 10}) survives the repository's existing realistic transaction-cost model,
judged against a matched coin-flip null run under the same costs.

Why this is decision-relevant: the momentum admission (activations 37318950814,
37361000967) was produced with ``BacktestConfig()`` — zero commission, zero
slippage. Momentum rebalances daily and every sign flip pays commission and
slippage on the full notional swing, so cost drag is a first-order qualification
that has never been measured on this universe. If realistic costs erase the edge,
the admission is wrong; if the edge survives, the admission is materially
stronger.

Cost model (predeclared; copied verbatim from the repository's established
"realistic" model in ``examples/ma_crossover_real_data.py`` and
``examples/volatility_regime_filter.py`` — not invented here):

    commission_per_trade   = 2.0   # $2.00 flat per fill
    commission_per_share   = 0.003 # $0.003 per share (0.3 cents)
    slippage_cents         = 2.0   # 2 cents per share fixed
    slippage_proportional  = 0.0005  # 5 bps proportional

Predeclared cost grid (three points, fixed before any result was observed):

    zero       : the reference (BacktestConfig()); must reproduce the prior
                 zero-cost lookback sweep exactly (cross-consistency gate).
    realistic  : the repository's realistic model above.
    stress2x   : exactly 2x every cost parameter — a bounded conservative upper
                 bound, not a tuned value.

Matched null: for every (lookback, cost level, asset) the coin-flip null is run
through ``noise_benchmark`` with the SAME ``cfg`` as the candidate, so both pay
the same cost model. Cost drag is therefore separated into (a) the candidate's
own drag and (b) the null's drag; the decision-relevant edge is the candidate's
absolute net medians plus the candidate-minus-null gap.

A-priori falsification prediction (declared before any cost result was seen):

1. Candidate absolute net medians decline monotonically across
   zero -> realistic -> stress2x (daily rebalancing pays cost on every flip).
2. The matched coin-flip null declines too, and by MORE than the candidate,
   because the iid {-1, 0, +1} null changes position on ~2/3 of bars while
   momentum changes far less often; so the candidate-minus-null gap may widen
   even as absolute returns fall.
3. Primary classification at the realistic level: the assets with a zero-cost
   positive REGIME_STABLE edge (AAPL, MSFT, GOOGL, AMZN, META, TSLA) retain it
   (SURVIVES); the assets without one (NVDA, JPM, JNJ, XOM) stay NO_EDGE.
   Falsifiable form: NO asset with a zero-cost positive edge is reclassified
   ERODED at the realistic level. If any is, the prediction is falsified and
   the admitted candidate is materially weakened.

Classification (per asset, per cost level), using only predeclared rules:
    positive_edge(lookback) := verdict == REGIME_STABLE and every segment
        median > 0 and mean(candidate medians) > mean(null medians).
    SURVIVES : every lookback positive at zero is also positive at this cost.
    ERODED   : at least one lookback positive at zero is not positive here.
    NO_EDGE  : no lookback is positive at zero (nothing to erode).

Signal-validity gate (cost-independent): a cost-sensitivity result is only
interpretable if the tested signal is actually the signal it claims to be.
The gate independently recomputes the true previous-``lookback``-day return sign
as ``log(close[i]) - log(close[i - lookback])`` and requires the signal's sign
to agree on >= 95% of bars. A signal that is +1 (or -1) on ~every bar is a
constant directional bet, not momentum, and its cost sensitivity says nothing
about a momentum edge. When the gate fails, the primary question is recorded as
UNVERIFIED_INVALID_SUBJECT rather than as a cost-robustness verdict.

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
INITIAL_CAPITAL = 1e6

# The repository's established realistic cost model (see module docstring).
REALISTIC_COST = dict(
    commission_per_trade=2.0,
    commission_per_share=0.003,
    slippage_cents=2.0,
    slippage_proportional=0.0005,
)

COST_LEVELS = ("zero", "realistic", "stress2x")
COST_CONFIGS = {
    "zero": {},
    "realistic": dict(REALISTIC_COST),
    "stress2x": {k: 2.0 * v for k, v in REALISTIC_COST.items()},
}

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
PRIOR_LOOKBACK_ARTIFACT = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"


def cost_config(level: str) -> dict:
    """Return the predeclared cost kwargs for one cost level."""
    return dict(COST_CONFIGS[level])


def segment_result(ticker, bars, lookback, level):
    """Walk-forward regime-stability result for one ticker/lookback/cost level.

    Uses the framework's own ``stress_segments`` (candidate) and
    ``noise_benchmark`` (matched null), both under the same ``cfg`` so the null
    pays the same costs as the candidate.
    """
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError(f"{ticker}: signals {len(sig)} != bars {len(closes)}")
    cfg = bt.BacktestConfig(initial_capital=INITIAL_CAPITAL, **cost_config(level))
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
        medians=[round(s.baseline_median_log_return, 3) for s in res.scenarios],
        null_medians=[round(s.noise_median_log_return, 3) for s in res.scenarios],
        candidate_dispersion=round(res.candidate_dispersion, 3),
        null_dispersion=round(res.null_dispersion, 3),
        n_folds=res.n_folds,
        verdict=res.overall_verdict,
    )


def mean_gap(sr):
    """Mean candidate median minus mean matched-null median (edge over null)."""
    return float(np.mean(sr["medians"]) - np.mean(sr["null_medians"]))


# Minimum fraction of bars on which the tested signal's sign must agree with the
# true previous-`lookback`-day return sign for the subject to count as momentum.
SIGNAL_AGREEMENT_MIN = 0.95


def signal_validity(signals, closes, lookback):
    """Validity gate for the tested signal (cost-independent).

    A genuine "long the previous ``lookback``-day return" signal must have the
    same sign as the true previous-``lookback``-day return on almost every bar.
    This recomputes that true return independently as
    ``log(close[i]) - log(close[i - lookback])`` and compares signs. It also
    reports the weight distribution and position-flip count, because a signal
    that is +1 on ~every bar (a constant directional bet) is not a momentum
    signal regardless of what its cost sensitivity looks like.
    """
    w = np.array([s.weight for s in signals], dtype=np.float64)
    n_pos = int(np.sum(w > 0))
    n_neg = int(np.sum(w < 0))
    n_zero = int(np.sum(w == 0))
    n_flips = int(np.sum(w[1:] != w[:-1]))
    active = n_pos + n_neg
    closes = np.asarray(closes, dtype=np.float64)
    true_sign = np.sign(np.log(closes[lookback:]) - np.log(closes[:-lookback]))
    sig_tail = w[lookback:]
    mask = (sig_tail != 0) & (true_sign != 0)
    agreement = float(np.mean(sig_tail[mask] == true_sign[mask])) if mask.any() else 0.0
    dominant_share = (max(n_pos, n_neg) / active) if active else 0.0
    degenerate = bool(dominant_share >= 0.99 or agreement < SIGNAL_AGREEMENT_MIN)
    return dict(
        n_pos=n_pos,
        n_neg=n_neg,
        n_zero=n_zero,
        n_flips=n_flips,
        dominant_weight_share=round(float(dominant_share), 4),
        true_return_sign_agreement=round(agreement, 4),
        degenerate=degenerate,
    )


def positive_edge(sr):
    """Predeclared 'positive edge' rule for one lookback result."""
    return (
        sr["verdict"] == "REGIME_STABLE"
        and all(m > 0 for m in sr["medians"])
        and mean_gap(sr) > 0
    )


def classify_cost_robustness(zero_profile, cost_profile):
    """Classify one asset's cost robustness from zero/cost per-lookback profiles.

    Returns a dict with the positive-lookback lists and the SURVIVES / ERODED /
    NO_EDGE classification for the given cost level.
    """
    zero_pos = [str(lb) for lb in LOOKBACKS if positive_edge(zero_profile[str(lb)])]
    cost_pos = [str(lb) for lb in LOOKBACKS if positive_edge(cost_profile[str(lb)])]
    if not zero_pos:
        cls = "NO_EDGE"
    elif all(lb in cost_pos for lb in zero_pos):
        cls = "SURVIVES"
    else:
        cls = "ERODED"
    return dict(
        zero_positive_lookbacks=zero_pos,
        cost_positive_lookbacks=cost_pos,
        classification=cls,
    )


def _mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def asset_cost_profile(zero_prof, cost_profiles):
    """Aggregate per-asset cost diagnostics across the cost grid.

    cost_profiles maps level -> {lookback: segment_result}. The zero-cost
    profile is ``zero_prof`` (same shape). Drag figures are candidate/null mean
    median changes relative to zero for each non-zero level.
    """
    out = {}
    for level in COST_LEVELS:
        prof = zero_prof if level == "zero" else cost_profiles[level]
        gaps = [mean_gap(prof[str(lb)]) for lb in LOOKBACKS]
        cand = [_mean(prof[str(lb)]["medians"]) for lb in LOOKBACKS]
        null = [_mean(prof[str(lb)]["null_medians"]) for lb in LOOKBACKS]
        out[level] = dict(
            mean_candidate_gap=round(_mean(gaps), 4),
            mean_candidate_median=round(_mean(cand), 4),
            mean_null_median=round(_mean(null), 4),
            positive_lookbacks=[str(lb) for lb in LOOKBACKS
                                if positive_edge(prof[str(lb)])],
        )
    for level in ("realistic", "stress2x"):
        out[level]["candidate_cost_drag"] = round(
            out[level]["mean_candidate_median"] - out["zero"]["mean_candidate_median"], 4
        )
        out[level]["null_cost_drag"] = round(
            out[level]["mean_null_median"] - out["zero"]["mean_null_median"], 4
        )
    out["classification_realistic"] = classify_cost_robustness(
        zero_prof, cost_profiles["realistic"]
    )
    out["classification_stress2x"] = classify_cost_robustness(
        zero_prof, cost_profiles["stress2x"]
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

    print("\n=== 3. Leakage review per asset (realistic cost) ===")
    target = ["AMZN", "JPM"]
    tickers = {}
    for ticker in target:
        bars, dates = bt.load_ticker(ticker)
        tickers[ticker] = bars
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=5)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        cfg = bt.BacktestConfig(initial_capital=INITIAL_CAPITAL, **REALISTIC_COST)
        res = bt.run_bars(list(bars), signals, cfg)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, INITIAL_CAPITAL)
        total_return = (res.equity_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
        print(f"  {ticker}: {bars.n_bars} bars, momentum signals {len(signals)}, "
              f"leakage [PASS], cost-filled total_return {total_return:+.3f}")
    print("  no-look-ahead: momentum sign at bar t uses closes[:t] only; "
          "the realistic-cost equity curve is fully explained by recorded fills.")

    print("\n=== 4. Cost grid (predeclared) ===")
    for level in COST_LEVELS:
        print(f"  {level:10s}: {cost_config(level)}")
    print("  walk-forward per segment train={}d / test={}d / warmup={}d / "
          "overlap={}d, min {} bars/segment, 4 contiguous volatility blocks, "
          "each lookback vs a matched coin-flip null under the SAME cost "
          "model\n".format(TRAIN, TEST, WARM, OVERLAP, MIN_SEGMENT_BARS))

    # Load the full universe once.
    all_tickers = {}
    for entry in manifest["entries"]:
        ticker = entry["ticker"]
        all_tickers[ticker] = bt.load_ticker(ticker)[0]
    ticker_order = list(all_tickers.keys())

    # results[level][ticker][lookback] = segment_result
    results = {
        level: {t: {lb: segment_result(t, all_tickers[t], lb, level) for lb in LOOKBACKS}
                for t in ticker_order}
        for level in COST_LEVELS
    }

    print("=== 5. Signal-validity gate (is the tested signal actually momentum?) ===")
    print("  A genuine momentum signal must match the true previous-N-day return")
    print("  sign on >= {:.0%} of bars; a near-constant +1/-1 position is a"
          " directional bet, not momentum.\n".format(SIGNAL_AGREEMENT_MIN))

    print("=== 6. Candidate vs matched null, per asset ===")
    per_asset = {}
    class_counts = {lvl: {"SURVIVES": 0, "ERODED": 0, "NO_EDGE": 0} for lvl in ("realistic", "stress2x")}
    lookback_edge_counts = {lvl: {str(lb): 0 for lb in LOOKBACKS} for lvl in COST_LEVELS}
    for ticker in ticker_order:
        zero_prof = {str(lb): results["zero"][ticker][lb] for lb in LOOKBACKS}
        cost_profiles = {
            lvl: {str(lb): results[lvl][ticker][lb] for lb in LOOKBACKS}
            for lvl in ("realistic", "stress2x")
        }
        agg = asset_cost_profile(zero_prof, cost_profiles)
        for lvl in ("realistic", "stress2x"):
            class_counts[lvl][agg[f"classification_{lvl}"]["classification"]] += 1
        for lvl in COST_LEVELS:
            for lb in LOOKBACKS:
                if positive_edge(results[lvl][ticker][lb]):
                    lookback_edge_counts[lvl][str(lb)] += 1

        # Signal-validity gate (cost-independent): compare the signal sign to
        # the independently recomputed true previous-N-day return sign.
        bars_t = all_tickers[ticker]
        closes_t = bars_t.closes_array()
        validity = {}
        for lb in LOOKBACKS:
            sig_t = bt.momentum_signals(closes_t, lookback=lb)
            validity[str(lb)] = signal_validity(sig_t, closes_t, lb)
        subject_valid_t = all(not validity[str(lb)]["degenerate"] for lb in LOOKBACKS)
        per_asset[ticker] = dict(
            zero=zero_prof,
            realistic=cost_profiles["realistic"],
            stress2x=cost_profiles["stress2x"],
            cost=agg,
            signal_validity=validity,
            subject_valid=subject_valid_t,
        )
        zr = agg["classification_realistic"]
        v5 = validity[str(5)]
        print("  {:6s} agree(true_ret_sign)={:.3f} dominant_share={:.3f} "
              "flips={:4d} subject_valid={:<5s} | zero_pos={:<10s} "
              "realistic={:<8s} stress2x={:<8s} gap(z/r/2x)={:+.3f}/{:+.3f}/{:+.3f}".format(
                  ticker,
                  v5["true_return_sign_agreement"],
                  v5["dominant_weight_share"],
                  v5["n_flips"],
                  str(subject_valid_t),
                  ",".join(agg["zero"]["positive_lookbacks"]) or "-",
                  zr["classification"],
                  agg["classification_stress2x"]["classification"],
                  agg["zero"]["mean_candidate_gap"],
                  agg["realistic"]["mean_candidate_gap"],
                  agg["stress2x"]["mean_candidate_gap"],
              ))
    print("")

    subject_valid = all(per_asset[t]["subject_valid"] for t in ticker_order)
    if not subject_valid:
        primary_verdict = "UNVERIFIED_INVALID_SUBJECT"
    else:
        eroded = class_counts["realistic"]["ERODED"]
        primary_verdict = "COST_FRAGILE" if eroded else "COST_ROBUST"
    print("subject_valid (all assets, all lookbacks): {}".format(subject_valid))
    print("primary_question_verdict: {}".format(primary_verdict))
    print("")

    # Determinism: rerun the target subset at the realistic level.
    res2 = {t: {lb: segment_result(t, tickers[t], lb, "realistic") for lb in LOOKBACKS}
            for t in target}
    out1 = json.dumps({t: results["realistic"][t] for t in target}, sort_keys=True)
    out2 = json.dumps(res2, sort_keys=True)
    print("determinism: realistic r1 == r2: {}".format(out1 == out2))
    assert out1 == out2, "non-deterministic output"

    # Zero-cost cross-consistency vs the prior lookback sweep (durable evidence).
    print("\n=== 7. Zero-cost column cross-consistency vs prior lookback sweep ===")
    prior = json.loads(PRIOR_LOOKBACK_ARTIFACT.read_text())
    cmismatch = False
    for ticker in ticker_order:
        for lb in LOOKBACKS:
            pub = prior["per_asset"][ticker][str(lb)]
            new = results["zero"][ticker][lb]
            ok = (pub["medians"] == new["medians"]
                  and pub["null_medians"] == new["null_medians"]
                  and pub["verdict"] == new["verdict"])
            if not ok:
                cmismatch = True
                print("  MISMATCH {} lb {}: prior {} vs new {}".format(
                    ticker, lb, pub, new))
    if cmismatch:
        print("  ZERO-COST CROSS-CONSISTENCY FAILED")
        sys.exit(1)
    print("  all 10 tickers x 3 lookbacks (medians/nulls/verdicts) -> MATCH")

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        initial_capital=INITIAL_CAPITAL,
        cost_levels=list(COST_LEVELS),
        cost_configs={lvl: cost_config(lvl) for lvl in COST_LEVELS},
        signal_validity_min_agreement=SIGNAL_AGREEMENT_MIN,
        subject_valid=subject_valid,
        primary_question_verdict=primary_verdict,
        per_asset=per_asset,
        universe=dict(
            classification_counts=class_counts,
            lookback_edge_counts=lookback_edge_counts,
            ticker_order=ticker_order,
        ),
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print("\nartifact written to: {}".format(ARTIFACT_PATH))
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity across the collected universe: "
          "exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())