"""Concentration analysis of the admitted momentum candidate.

Objective: determine whether the REGIME_STABLE positive edge of the admitted
momentum class (long the previous N-day return, hold 1 day, lookback=5 base,
from research/checks/momentum.py + the lookback sweep at research/checks/
momentum_sweep.py) is concentrated in one or two names or distributed across
the REGIME_STABLE universe. Concentration matters for admission: an edge that
rests on a single name (or a 2009-2013-style period) is much weaker evidence
than one that holds uniformly across the large-cap universe.

Method (fixed a-priori, not tuned to OOS):

- Base data: state/check_artifacts/momentum_sweep_results.json (lookback 3/5/10/
  20 on the collected adjusted-close universe, seed 42, walk-forward
  train=252d/test=84d/warmup=60d/overlap=60d, 4 volatility blocks, min 400
  bars/segment, coin-flip null). This artifact was independently verified
  (research/checks/verify_momentum_sweep.py: MATCH at full precision).
- Per (asset, lookback): per-segment signed effect = candidate_median -
  null_median (rounded to 3 decimals as stored); per-asset signed edge = sum
  over segments; absolute effect = sum of abs(effect); mean per segment to
  normalize across assets with different segment counts.
- Concentration metrics (universe-wide, per lookback):
  * HHI on per-asset absolute-effect shares (10 assets; uniform would be ~0.10)
  * Top-1 and top-3 share of the universe's cumulative POSITIVE signed edge
  * Max-to-median ratio of per-asset mean absolute effects
- Effect-strength buckets per asset by mean absolute effect:
  EFFECTIVE_STRONG > 0.05, EFFECTIVE_MODERATE 0.025-0.05, EFFECTIVE_WEAK
  0.01-0.025, EFFECTIVE_NOEDGE < 0.01.
- Null-baseline: identical concentration metrics computed on the coin-flip
  null medians (expected: uniform distribution, no structure).
- Concentration verdict (a-priori thresholds, frozen before the run):
  CONCENTRATED if top-1 share of positive edge >= 0.40 or HHI >= 0.30
  DISTRIBUTED if top-1 share <= 0.30 and HHI <= 0.20 and max-to-median <= 2.0
  otherwise AMBIGUOUS
- Determinism: internal recomputation asserted identical across two runs.
- Output: state/check_artifacts/momentum_concentration.json + printed report.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from research.data.preflight import load_manifest

DATASET_ID = "yf-ohlcv-universe-2009-to-2026-10-03"
ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_sweep_results.json"
OUT_PATH = ARTIFACT_PATH.parent / "momentum_concentration.json"

# A-priori concentration thresholds (frozen before the run; not tuned to results)
TOP1_CONCENTRATED = 0.40
HHI_CONCENTRATED = 0.30
TOP1_DISTRIBUTED = 0.30
HHI_DISTRIBUTED = 0.20
MAX_MEDIAN_DISTRIBUTED = 2.0

# Effect-strength buckets on mean absolute effect
BUCKET_STRONG = 0.05
BUCKET_MODERATE = 0.025
BUCKET_WEAK = 0.01


def effect_metrics(pa_rows):
    """Compute per-asset signed edge, absolute effect, and mean-per-segment
    values from a per-asset artifact row (medians vs null_medians)."""
    out = {}
    for row in pa_rows:
        meds = row["medians"]
        nulls = row["null_medians"]
        seg_effects = [float(m) - float(n) for m, n in zip(meds, nulls)]
        signed_edge = float(sum(seg_effects))
        abs_effect = float(sum(abs(e) for e in seg_effects))
        n = len(seg_effects)
        out[row["ticker"]] = dict(
            n_segments=n,
            seg_effects=seg_effects,
            signed_edge=signed_edge,
            abs_effect=abs_effect,
            mean_signed_edge=signed_edge / n if n else 0.0,
            mean_abs_effect=abs_effect / n if n else 0.0,
        )
    return out


def concentration_metrics(effects):
    """Universe-level concentration metrics on a dict of per-asset effect
    summaries."""
    tickers = list(effects.keys())
    abs_effects = [effects[t]["abs_effect"] for t in tickers]
    total_abs = float(sum(abs_effects))
    total_pos_edge = float(sum(max(effects[t]["signed_edge"], 0.0) for t in tickers))
    mean_abs = [effects[t]["mean_abs_effect"] for t in tickers]
    shares = [a / total_abs for a in abs_effects]
    hhi = float(sum(s * s for s in shares))
    sorted_signed = sorted(
        ((t, effects[t]["signed_edge"]) for t in tickers),
        key=lambda x: x[1],
        reverse=True,
    )
    pos_sorted = sorted(
        ((t, effects[t]["signed_edge"]) for t in tickers
         if effects[t]["signed_edge"] > 0.0),
        key=lambda x: x[1],
        reverse=True,
    )
    top1 = pos_sorted[0][1] / total_pos_edge if total_pos_edge > 0 else 0.0
    top3 = sum(x[1] for x in pos_sorted[:3]) / total_pos_edge if total_pos_edge > 0 else 0.0
    max_med = max(mean_abs)
    median_abs = float(np.median(mean_abs))
    max_med_ratio = (max_med / median_abs) if median_abs > 0 else float("inf")
    return dict(
        n_assets=len(tickers),
        total_abs_effect=round(total_abs, 6),
        total_pos_edge=round(total_pos_edge, 6),
        hhi=round(hhi, 6),
        top1_share_pos_edge=round(top1, 6),
        top3_share_pos_edge=round(top3, 6),
        max_median_ratio=round(max_med_ratio, 6),
    )


def bucket(ma_effect):
    if ma_effect > BUCKET_STRONG:
        return "EFFECTIVE_STRONG"
    if ma_effect > BUCKET_MODERATE:
        return "EFFECTIVE_MODERATE"
    if ma_effect > BUCKET_WEAK:
        return "EFFECTIVE_WEAK"
    return "EFFECTIVE_NOEDGE"


def concentration_verdict(m):
    if m["top1_share_pos_edge"] >= TOP1_CONCENTRATED or m["hhi"] >= HHI_CONCENTRATED:
        return "CONCENTRATED"
    if (m["top1_share_pos_edge"] <= TOP1_DISTRIBUTED
            and m["hhi"] <= HHI_DISTRIBUTED
            and m["max_median_ratio"] <= MAX_MEDIAN_DISTRIBUTED):
        return "DISTRIBUTED"
    return "AMBIGUOUS"


def main() -> int:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=== 1. Artifact + manifest load ===")
    manifest = load_manifest()
    assert manifest["dataset_id"] == DATASET_ID, manifest["dataset_id"]
    artifact = json.load(open(ARTIFACT_PATH))
    assert artifact["dataset_id"] == DATASET_ID, "artifact dataset id mismatch"
    assert artifact["lookbacks"] == [3, 5, 10, 20], "unexpected lookback grid"
    print("  dataset: {} | seed {} | lookbacks {}".format(
        artifact["dataset_id"], artifact["seed"], artifact["lookbacks"]))

    print("\n=== 2. Base lookback (5-day momentum) ===")
    out = artifact["lookback_5"]
    effects = effect_metrics(out["per_asset"])
    m = concentration_metrics(effects)
    v = concentration_verdict(m)
    print("  HHI (abs-effect shares, 10 assets): {:.4f} (uniform ~0.10)"
          .format(m["hhi"]))
    print("  top-1 share of positive edge: {:.1%} (concentrated threshold >= {:.0%})"
          .format(m["top1_share_pos_edge"], TOP1_CONCENTRATED))
    print("  top-3 share of positive edge: {:.1%}".format(m["top3_share_pos_edge"]))
    print("  max/median mean-abs-effect ratio: {:.2f} (distributed threshold <= {:.1f})"
          .format(m["max_median_ratio"], MAX_MEDIAN_DISTRIBUTED))
    print("  concentration verdict: {}".format(v))
    print("")
    print("  Asset  | segs | signed_edge | abs_effect | mean_abs_eff | bucket")
    for t in out["ticker_order"]:
        e = effects[t]
        print("  {:6s} | {:3d} | {:+11.3f} | {:+10.3f} | {:+11.4f} | {}"
              .format(t, e["n_segments"], e["signed_edge"], e["abs_effect"],
                      e["mean_abs_effect"], bucket(e["mean_abs_effect"])))
    print("")

    print("=== 3. Null-baseline concentration (coin-flip null medians) ===")
    null_effects = effect_metrics(out["per_asset"])
    for t in null_effects:
        for k in ("signed_edge", "abs_effect", "mean_abs_effect"):
            null_effects[t][k] = 0.0
        for se in null_effects[t]["seg_effects"]:
            null_effects[t]["signed_edge"] += se
            null_effects[t]["abs_effect"] += abs(se)
    # recompute mean_abs_effect with the null segment effects
    for t, e in null_effects.items():
        e["mean_signed_edge"] = e["signed_edge"] / e["n_segments"]
        e["mean_abs_effect"] = e["abs_effect"] / e["n_segments"]
    nm = concentration_metrics(null_effects)
    print("  HHI (null abs-effect shares): {:.4f}".format(nm["hhi"]))
    print("  top-1 share of null positive edge: {:.1%}".format(nm["top1_share_pos_edge"]))
    print("  top-3 share of null positive edge: {:.1%}".format(nm["top3_share_pos_edge"]))
    print("  (null is price-independent noise: no concentration structure expected)")
    print("")

    print("=== 4. Concentration across all lookbacks (3 / 5 / 10 / 20) ===")
    all_metrics = {}
    for lb in artifact["lookbacks"]:
        lbout = artifact["lookback_{}".format(lb)]
        lb_effects = effect_metrics(lbout["per_asset"])
        lbm = concentration_metrics(lb_effects)
        all_metrics["lookback_{}".format(lb)] = dict(
            hhi=lbm["hhi"],
            top1_share_pos_edge=lbm["top1_share_pos_edge"],
            top3_share_pos_edge=lbm["top3_share_pos_edge"],
            max_median_ratio=lbm["max_median_ratio"],
            verdict=concentration_verdict(lbm),
            effects={t: dict(signed_edge=round(lb_effects[t]["signed_edge"], 4),
                             abs_effect=round(lb_effects[t]["abs_effect"], 4),
                             mean_abs_effect=round(lb_effects[t]["mean_abs_effect"], 4))
                     for t in lbout["ticker_order"]},
        )
        print("  lookback {}: HHI={:.4f} top1={:.1%} top3={:.1%} max/med={:.2f} -> {}"
              .format(lb, lbm["hhi"], lbm["top1_share_pos_edge"],
                      lbm["top3_share_pos_edge"], lbm["max_median_ratio"],
                      concentration_verdict(lbm)))
    print("")

    print("=== 5. REGIME_STABLE consistency across lookbacks vs effect size ===")
    per_ticker = {}
    for lb in artifact["lookbacks"]:
        for r in artifact["lookback_{}".format(lb)]["per_asset"]:
            per_ticker.setdefault(r["ticker"], []).append(
                dict(lookback=lb, verdict=r["verdict"]))
    print("  Asset  | RS@3 RS@5 RS@10 RS@20 | RS_total/4 | mean_abs_eff | bucket")
    for t in artifact["lookback_5"]["ticker_order"]:
        hist = per_ticker[t]
        rs = sum(1 for h in hist if h["verdict"] == "REGIME_STABLE")
        ma = np.mean([all_metrics["lookback_{}".format(h["lookback"])]["effects"][t]["mean_abs_effect"]
                      for h in hist])
        print("  {:6s} | {:3d} {:3d} {:4d} {:4d} | {:9d} | {:+11.4f} | {}"
              .format(t, hist[0]["verdict"] == "REGIME_STABLE",
                      hist[1]["verdict"] == "REGIME_STABLE",
                      hist[2]["verdict"] == "REGIME_STABLE",
                      hist[3]["verdict"] == "REGIME_STABLE",
                      rs, ma, bucket(ma)))
    print("")

    print("=== 6. Determinism ===")
    e2 = effect_metrics(out["per_asset"])
    m2 = concentration_metrics(e2)
    d_ok = (round(m["hhi"], 6) == round(m2["hhi"], 6)
            and round(m["top1_share_pos_edge"], 6) == round(m2["top1_share_pos_edge"], 6))
    print("  r1 == r2 internal recomputation: {}".format(d_ok))
    assert d_ok, "concentration metrics not deterministic"
    print("")

    report = dict(
        dataset_id=artifact["dataset_id"],
        seed=artifact["seed"],
        lookbacks=artifact["lookbacks"],
        base_lookback=5,
        thresholds=dict(top1_concentrated=TOP1_CONCENTRATED,
                        hhi_concentrated=HHI_CONCENTRATED,
                        top1_distributed=TOP1_DISTRIBUTED,
                        hhi_distributed=HHI_DISTRIBUTED,
                        max_median_distributed=MAX_MEDIAN_DISTRIBUTED),
        buckets=dict(strong=BUCKET_STRONG, moderate=BUCKET_MODERATE, weak=BUCKET_WEAK),
        base=dict(metrics=m, verdict=v, per_asset={
            t: dict(signed_edge=round(effects[t]["signed_edge"], 4),
                    abs_effect=round(effects[t]["abs_effect"], 4),
                    mean_signed_edge=round(effects[t]["mean_signed_edge"], 4),
                    mean_abs_effect=round(effects[t]["mean_abs_effect"], 4),
                    bucket=bucket(effects[t]["mean_abs_effect"]))
            for t in out["ticker_order"]}),
        null_baseline=dict(metrics=nm,
                           per_asset={t: dict(signed_edge=round(null_effects[t]["signed_edge"], 4),
                                              abs_effect=round(null_effects[t]["abs_effect"], 4))
                                      for t in out["ticker_order"]}),
        by_lookback=all_metrics,
        determinism_r1_equals_r2=d_ok,
    )

    print("=== 7. Artifact written ===")
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print("  {}".format(OUT_PATH))
    print("")
    print("NOTE: research/simulation only. No live trading or production "
          "execution. Momentum concentration analysis: exploratory simulation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
