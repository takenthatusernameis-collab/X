"""Real-data cost-sensitivity check for short-horizon momentum.

Objective (focused task R-001): run a bounded cost-sensitivity check for
lookback 3 / 5 / 10 momentum under the repository's existing realistic cost
model and a matched coin-flip null, and classify cost robustness.

Why this check exists
---------------------
`state/LEARNING_STATE.md` admitted the lookback-5 momentum cell as candidate
positive evidence (activation 37318950814) and qualified it with a real-data
lookback sweep (37361000967). That evidence was produced with zero transaction
costs, so its decision-relevance depends on whether the edge survives
execution costs. This check answers that question on the existing 10-asset
collected universe with the existing walk-forward / regime / coin-flip-null
machinery.

Signal specification
--------------------
While preparing this check, the framework's ``momentum_signals`` was found to
average log *prices* over the lookback window instead of the lookback return:

    ret = mean(log(closes[i - lookback + 1 : i + 1]))      # defective

Because ``log(price) > 0`` whenever price > 1, that expression is positive on
essentially every bar, so the "momentum" signal was a constant long position
(see the ``signal_diagnostics`` block in the artifact, and the corrected
implementation in ``research/backtest/regime_stability.py``). The check below
therefore uses the corrected signal, defined as the sign of the cumulative log
return over the lookback window:

    ret = log(closes[i]) - log(closes[i - lookback])

Predeclared cost grid (fixed before execution; not tuned after results)
----------------------------------------------------------------------
    zero       : commission_per_trade=0, commission_per_share=0,
                 slippage_cents=0, slippage_proportional=0
    realistic  : commission_per_trade=2.0, commission_per_share=0.003,
                 slippage_cents=2.0, slippage_proportional=0.0005
                 (the repository's existing realistic cost model, from
                 examples/ma_crossover_real_data.py:131-135)
    stress_2x  : every realistic component doubled

Each cell is run through the same walk-forward regime gate used by the prior
momentum cells, and the candidate is compared against a coin-flip null run on
the same segments under the same cost configuration (the matched null).

A-priori prediction (recorded before execution)
-----------------------------------------------
P1 edge survival : fewer than half of the assets that show a positive
                   zero-cost REGIME_STABLE edge at lookback 5 will retain such
                   an edge at realistic costs.
P2 median decline: the mean per-asset segment median across zero-cost-admitted
                   assets declines by >= 40% from zero to realistic costs.
P3 monotonicity  : for every asset, realistic-cost medians are >= stress_2x
                   medians (higher costs cannot improve a cost-paying backtest).
P4 lookback      : the corrected signal produces materially different candidate
                   medians across lookbacks 3 / 5 / 10 (max pairwise absolute
                   difference > 0 for at least one asset), unlike the defective
                   price-level signal, which was lookback-invariant.

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
CANONICAL_LOOKBACK = 5

ARTIFACT_DIR = Path.cwd() / "state" / "check_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "momentum_cost_sensitivity_results.json"
LEGACY_SWEEP_PATH = ARTIFACT_DIR / "momentum_lookback_sweep_results.json"

COST_GRID = [
    ("zero", dict(commission_per_trade=0.0, commission_per_share=0.0,
                  slippage_cents=0.0, slippage_proportional=0.0)),
    ("realistic", dict(commission_per_trade=2.0, commission_per_share=0.003,
                       slippage_cents=2.0, slippage_proportional=0.0005)),
    ("stress_2x", dict(commission_per_trade=4.0, commission_per_share=0.006,
                       slippage_cents=4.0, slippage_proportional=0.001)),
]
COST_MODEL_SOURCE = (
    "examples/ma_crossover_real_data.py:131-135 (commission_per_trade=2.0, "
    "commission_per_share=0.003, slippage_cents=2.0, slippage_proportional=0.0005); "
    "zero and stress_2x are the declared bounds of this bounded sensitivity grid"
)

A_PRIORI_PREDICTION = {
    "P1_edge_survival": (
        "Fewer than half of the assets with a positive zero-cost REGIME_STABLE "
        "edge at lookback 5 retain such an edge at realistic costs."
    ),
    "P2_median_decline": (
        "The mean per-asset segment median across zero-cost-admitted assets "
        "declines by at least 40% from zero to realistic costs."
    ),
    "P3_monotonicity": (
        "For every asset, realistic-cost medians are >= stress_2x medians."
    ),
    "P4_lookback_dependence": (
        "Corrected-signal candidate medians differ across lookbacks 3/5/10 "
        "(max pairwise absolute difference > 0 for at least one asset)."
    ),
}


def cost_cfg(name: str) -> bt.BacktestConfig:
    params = dict(COST_GRID)[name]
    return bt.BacktestConfig(initial_capital=1e6, warmup_periods=0, **params)


def segment_result(bars, lookback: int, cfg: bt.BacktestConfig) -> dict:
    """One asset / lookback / cost cell through the existing regime gate."""
    closes = bars.closes_array()
    labels = bt.volatility_blocks(closes, n_blocks=N_BLOCKS, window=WINDOW)
    seg_fn = bt.segment_fn_from_labels(labels)
    sig = bt.momentum_signals(closes, lookback=lookback)
    if len(sig) != len(closes):
        raise ValueError("signal length mismatch")
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
        cfg=cfg,
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


def edge_at(cell: dict) -> bool:
    """A positive REGIME_STABLE edge: uniformly positive segment medians."""
    return cell["verdict"] == "REGIME_STABLE" and all(m > 0 for m in cell["medians"])


def classify(zero: dict, realistic: dict, stress: dict) -> str:
    """Predeclared cost-robustness classification for one asset."""
    ez, er, es = edge_at(zero), edge_at(realistic), edge_at(stress)
    if not ez:
        return "NO_EDGE"
    if er and es:
        return "COST_ROBUST"
    if er and not es:
        return "COST_MARGINAL"
    return "COST_FRAGILE"


def universe_verdict(class_counts: dict) -> str:
    """Predeclared universe-level cost-robustness verdict."""
    survive = class_counts.get("COST_ROBUST", 0) + class_counts.get("COST_MARGINAL", 0)
    if survive >= 6:
        return "SURVIVES"
    if survive >= 3:
        return "MARGINAL"
    return "FAILS"


def legacy_long_fraction(closes, lookback: int) -> float:
    """Fraction of post-warmup bars the defective price-level rule went long.

    This reproduces the pre-repair ``momentum_signals`` expression so the
    artifact documents the defect numerically.
    """
    n = len(closes)
    weights = np.zeros(n, dtype=np.float64)
    for i in range(int(lookback), n):
        ret = float(np.mean(np.log(closes[i - int(lookback) + 1 : i + 1])))
        weights[i] = 1.0 if ret > 0 else -1.0
    nz = weights != 0
    return float(np.mean(weights[nz] > 0)) if nz.any() else 0.0


def signal_diagnostics(bars) -> dict:
    closes = bars.closes_array()
    out = {}
    for lb in LOOKBACKS:
        corrected = np.array([s.weight for s in bt.momentum_signals(closes, lb)])
        nz = corrected != 0
        out[str(lb)] = {
            "legacy_long_fraction": round(legacy_long_fraction(closes, lb), 4),
            "corrected_long_fraction": round(float(np.mean(corrected[nz] > 0)), 4),
        }
    lb_a, lb_b = 3, 10
    a = np.array([s.weight for s in bt.momentum_signals(closes, lb_a)])
    b = np.array([s.weight for s in bt.momentum_signals(closes, lb_b)])
    both = (a != 0) & (b != 0)
    out["corrected_lookback_sign_agreement_3_vs_10"] = round(
        float(np.mean(a[both] == b[both])), 4)
    return out


def max_pairwise_median_diff(cells: dict) -> float:
    """Largest absolute difference in candidate medians across lookbacks."""
    lb_list = sorted(cells.keys(), key=int)
    vecs = [np.array(cells[lb]["medians"], dtype=np.float64) for lb in lb_list]
    best = 0.0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if len(vecs[i]) == len(vecs[j]):
                best = max(best, float(np.max(np.abs(vecs[i] - vecs[j]))))
    return round(best, 3)


def mean_edge(levels: dict, level: str) -> float:
    """Mean of the per-segment candidate medians at one cost level."""
    return float(np.mean(levels[level][str(CANONICAL_LOOKBACK)]["medians"]))


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

    print("\n=== 3. Leakage review (corrected momentum signal, realistic costs) ===")
    leak_cfg = cost_cfg("realistic")
    for ticker in ("AMZN", "JPM"):
        bars, _ = bt.load_ticker(ticker)
        closes = bars.closes_array()
        signals = bt.momentum_signals(closes, lookback=CANONICAL_LOOKBACK)
        bt.check_signal_integrity(signals, [b.date for b in bars], warmup=0)
        res = bt.run_bars(list(bars), signals, leak_cfg)
        bt.check_equity_matches_fills(res.equity_curve, res.trades, closes, 1e6)
        total_return = (res.equity_curve[-1] - 1e6) / 1e6
        print(f"  {ticker}: {bars.n_bars} bars, signals {len(signals)}, "
              f"leakage [PASS], total_return {total_return:+.3f}")
    print("  no-look-ahead: signal at bar t uses closes[:t+1] only; neutral bars "
          "before the lookback window carry equity forward.")

    tickers = {}
    for entry in manifest["entries"]:
        tickers[entry["ticker"]] = bt.load_ticker(entry["ticker"])[0]
    ticker_order = list(tickers.keys())

    print("\n=== 4. Signal diagnostics (defect documentation) ===")
    diagnostics = {t: signal_diagnostics(tickers[t]) for t in ticker_order}
    for t in ticker_order:
        d = diagnostics[t]
        print(f"  {t:6s} legacy long fraction {d['5']['legacy_long_fraction']:.3f} "
              f"| corrected long fraction {d['5']['corrected_long_fraction']:.3f} "
              f"| corrected lb3-vs-lb10 sign agreement "
              f"{d['corrected_lookback_sign_agreement_3_vs_10']:.3f}")

    print("\n=== 5. Cost grid (predeclared) ===")
    for name, params in COST_GRID:
        print(f"  {name}: {params}")
    print(f"  source: {COST_MODEL_SOURCE}")

    print("\n=== 6. Grid evaluation: 10 assets x 3 lookbacks x 3 cost levels ===")
    per_asset = {}
    for ticker in ticker_order:
        levels = {}
        for level, _ in COST_GRID:
            cfg = cost_cfg(level)
            levels[level] = {
                str(lb): segment_result(tickers[ticker], lb, cfg) for lb in LOOKBACKS
            }
        cls = classify(levels["zero"][str(CANONICAL_LOOKBACK)],
                       levels["realistic"][str(CANONICAL_LOOKBACK)],
                       levels["stress_2x"][str(CANONICAL_LOOKBACK)])
        per_asset[ticker] = {
            "cost_robustness": cls,
            "edge": {
                level: edge_at(levels[level][str(CANONICAL_LOOKBACK)])
                for level, _ in COST_GRID
            },
            "levels": levels,
            "lookback_median_max_abs_diff": {
                level: max_pairwise_median_diff(levels[level]) for level, _ in COST_GRID
            },
        }
        e = per_asset[ticker]["edge"]
        print(f"  {ticker:6s} class={cls:13s} edge zero={e['zero']} "
              f"real={e['realistic']} stress={e['stress_2x']} "
              f"lb_max_diff={per_asset[ticker]['lookback_median_max_abs_diff']}")

    class_counts = {"COST_ROBUST": 0, "COST_MARGINAL": 0, "COST_FRAGILE": 0,
                    "NO_EDGE": 0}
    for t in ticker_order:
        class_counts[per_asset[t]["cost_robustness"]] += 1
    verdict = universe_verdict(class_counts)
    admitted = [t for t in ticker_order if per_asset[t]["edge"]["zero"]]
    mean_edges = {
        level: round(float(np.mean([
            mean_edge(per_asset[t]["levels"], level) for t in admitted])),
            4) if admitted else None
        for level, _ in COST_GRID
    }
    decline = None
    if admitted and mean_edges["zero"] not in (None, 0.0):
        decline = round(
            (mean_edges["zero"] - mean_edges["realistic"]) / mean_edges["zero"], 4)

    print("\n=== 7. Classification summary ===")
    print(f"  class counts: {class_counts}")
    print(f"  zero-cost admitted assets: {admitted}")
    print(f"  n_survive_realistic: {class_counts['COST_ROBUST'] + class_counts['COST_MARGINAL']}")
    print(f"  mean edge by level: {mean_edges}")
    print(f"  median-edge decline (zero -> realistic): {decline}")
    print(f"  universe cost-robustness verdict: {verdict}")

    # P1: fewer than half of zero-cost-admitted assets retain the edge.
    retain = sum(1 for t in admitted if per_asset[t]["edge"]["realistic"])
    p1_supported = admitted and retain < len(admitted) / 2.0
    # P2: decline >= 40%.
    p2_supported = decline is not None and decline >= 0.40
    # P3: monotone non-improvement for every asset at the canonical lookback.
    p3_supported = all(
        all(r >= s for r, s in zip(
            per_asset[t]["levels"]["realistic"][str(CANONICAL_LOOKBACK)]["medians"],
            per_asset[t]["levels"]["stress_2x"][str(CANONICAL_LOOKBACK)]["medians"]))
        for t in ticker_order
    )
    # P4: at least one asset shows lookback-dependent corrected medians.
    p4_supported = any(
        per_asset[t]["lookback_median_max_abs_diff"]["zero"] > 0 for t in ticker_order)

    prediction_assessment = {
        "P1_edge_survival": {
            "predicted": A_PRIORI_PREDICTION["P1_edge_survival"],
            "observed": f"{retain}/{len(admitted)} zero-cost-admitted assets "
                        f"retain a positive REGIME_STABLE edge at realistic costs",
            "supported": bool(p1_supported),
        },
        "P2_median_decline": {
            "predicted": A_PRIORI_PREDICTION["P2_median_decline"],
            "observed": f"mean-edge decline zero -> realistic = {decline}",
            "supported": bool(p2_supported),
        },
        "P3_monotonicity": {
            "predicted": A_PRIORI_PREDICTION["P3_monotonicity"],
            "observed": "realistic medians >= stress_2x medians for all assets"
                        if p3_supported else
                        "at least one asset has realistic medians < stress_2x medians",
            "supported": bool(p3_supported),
        },
        "P4_lookback_dependence": {
            "predicted": A_PRIORI_PREDICTION["P4_lookback_dependence"],
            "observed": "max |median difference| across lookbacks (zero cost) = "
                        f"{max(per_asset[t]['lookback_median_max_abs_diff']['zero'] for t in ticker_order)}",
            "supported": bool(p4_supported),
        },
    }
    print("\n=== 8. A-priori prediction assessment ===")
    for k, v in prediction_assessment.items():
        print(f"  {k}: supported={v['supported']} | {v['observed']}")

    # Zero-cost cross-check against the legacy lookback-sweep artifact: the
    # legacy artifact was produced by the defective price-level signal, so it is
    # NOT expected to match the corrected zero-cost column. The max absolute
    # difference is recorded as evidence of the defect's impact.
    legacy_diff = None
    if LEGACY_SWEEP_PATH.exists():
        legacy = json.loads(LEGACY_SWEEP_PATH.read_text())
        diffs = []
        for t in ticker_order:
            for lb in LOOKBACKS:
                pub = legacy["universe"]["per_asset"][t]["lookback_medians"][str(lb)]
                new = per_asset[t]["levels"]["zero"][str(lb)]["medians"]
                if len(pub) == len(new):
                    diffs.extend(abs(a - b) for a, b in zip(pub, new))
        legacy_diff = round(max(diffs), 3) if diffs else None

    # Determinism: rerun the AMZN realistic lookback-5 cell and assert equality.
    det1 = segment_result(tickers["AMZN"], CANONICAL_LOOKBACK, cost_cfg("realistic"))
    det2 = segment_result(tickers["AMZN"], CANONICAL_LOOKBACK, cost_cfg("realistic"))
    determinism = json.dumps(det1, sort_keys=True) == json.dumps(det2, sort_keys=True)
    print(f"\n  determinism (AMZN realistic lookback 5 r1 == r2): {determinism}")
    assert determinism, "non-deterministic output"

    artifact = dict(
        dataset_id=DATASET_ID,
        seed=SEED,
        lookbacks=list(LOOKBACKS),
        canonical_lookback=CANONICAL_LOOKBACK,
        cost_levels=[name for name, _ in COST_GRID],
        cost_grid={name: params for name, params in COST_GRID},
        cost_model_source=COST_MODEL_SOURCE,
        train=TRAIN, test=TEST, warmup=WARM, overlap=OVERLAP,
        window=WINDOW, n_blocks=N_BLOCKS, min_segment_bars=MIN_SEGMENT_BARS,
        signal_spec="sign(log(closes[i]) - log(closes[i - lookback]))",
        a_priori_prediction=A_PRIORI_PREDICTION,
        signal_diagnostics=diagnostics,
        per_asset=per_asset,
        universe=dict(
            ticker_order=ticker_order,
            class_counts=class_counts,
            n_survive_realistic=(class_counts["COST_ROBUST"]
                                 + class_counts["COST_MARGINAL"]),
            cost_robustness_verdict=verdict,
            zero_cost_admitted_assets=admitted,
            mean_edge_by_level=mean_edges,
            mean_edge_decline_zero_to_realistic=decline,
        ),
        prediction_assessment=prediction_assessment,
        legacy_zero_cost_cross_check=dict(
            legacy_artifact=LEGACY_SWEEP_PATH.name,
            max_abs_median_diff_vs_corrected=legacy_diff,
            note=("the legacy lookback-sweep artifact was produced by the defective "
                  "price-level signal; a large difference is expected and is "
                  "evidence that the prior momentum result did not measure a "
                  "momentum signal"),
        ),
        determinism=determinism,
    )
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nartifact written to: {ARTIFACT_PATH}")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum cost-sensitivity check: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())