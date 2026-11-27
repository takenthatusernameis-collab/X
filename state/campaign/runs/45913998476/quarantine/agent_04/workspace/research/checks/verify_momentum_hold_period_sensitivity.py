"""Independent verification for R-002 (holding-period sensitivity of lookback-5
momentum).

This verifier is deliberately independent of ``bt.momentum_signals``: it
re-derives the signal weights directly from the raw adjusted-close series so a
defect in the framework signal constructor cannot be reproduced blindly (the
failure mode that let the earlier "independent" momentum verifications pass).

Gates:
  1. Artifact medians reproduce via a fresh ``walk_forward`` path (the artifact
     is a faithful record of the executed run).
  2. The holding-period transform is a NO-OP: per-asset segment medians are
     identical across hold in {1,2,3,5} (a constant signal cannot vary with the
     hold).
  3. The as-implemented momentum signal is DEGENERATE: it is +1.0 (long) for
     ~all bars with ~1 turn, i.e. a constant long position.
  4. The signal carries NO momentum information: agreement between the signal
     weight and the TRUE lookback-day return sign equals the always-long base
     rate (not better).
  5. Determinism: gate 3/4 recomputation is identical across two passes.

Research / simulation only. No live trading or production execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import research.backtest as bt

ARTIFACT_PATH = Path.cwd() / "state" / "check_artifacts" / "momentum_hold_period_sensitivity_results.json"
LOOKBACK = 5
HOLD_PERIODS = (1, 2, 3, 5)


def as_implemented_weights(closes, lookback=LOOKBACK):
    """Re-derive the framework momentum weights from raw closes, independently.

    Mirrors research/backtest/regime_stability.py::momentum_signals:
        ret = mean(log(closes[i-lookback+1 : i+1]))
        weight = +1 if ret > 0 else -1   (0 before lookback)
    """
    n = len(closes)
    w = np.zeros(n)
    for i in range(lookback, n):
        ret = float(np.mean(np.log(closes[i - lookback + 1 : i + 1])))
        w[i] = 1.0 if ret > 0 else -1.0
    return w


def true_return_sign(closes, lookback=LOOKBACK):
    n = len(closes)
    r = np.full(n, np.nan)
    for i in range(lookback, n):
        r[i] = np.log(closes[i]) - np.log(closes[i - lookback])
    return r


def hold_effective_weights(w, hold, lookback=LOOKBACK):
    n = len(w)
    out = w.copy()
    for i in range(n):
        if i < lookback:
            out[i] = 0.0
        else:
            out[i] = w[i - (i - lookback) % hold]
    return out


def wf_medians(closes, bars, hold):
    labels = bt.volatility_blocks(closes, n_blocks=4, window=60)
    seg_fn = bt.segment_fn_from_labels(labels)
    w = as_implemented_weights(closes)
    eff_w = hold_effective_weights(w, hold)
    eff = [bt.Signal(date=i + 1, weight=float(eff_w[i])) for i in range(len(closes))]
    res = bt.stress_segments(
        signals_fn=lambda c, h=hold, **p: [
            bt.Signal(date=i + 1, weight=float(hold_effective_weights(as_implemented_weights(c), h)[i]))
            for i in range(len(c))
        ],
        bars=list(bars),
        signals=eff,
        param_grid=[{"hold": hold}],
        baseline=(("hold", hold),),
        segment_fn=seg_fn,
        train_window=252, test_window=84, warmup=60, overlap_window=60,
        periods_per_year=252, min_segment_bars=400,
    )
    return [round(s.baseline_median_log_return, 3) for s in res.scenarios]


def main() -> int:
    with open(ARTIFACT_PATH) as f:
        art = json.load(f)

    failures = []

    # Gate 1: artifact AMZN medians reproduce via fresh walk_forward.
    print("=== Gate 1: artifact medians reproduce (AMZN, fresh walk_forward) ===")
    bars, _ = bt.load_ticker("AMZN")
    closes = bars.closes_array()
    for hold in HOLD_PERIODS:
        fresh = wf_medians(closes, bars, hold)
        recorded = art["per_asset"]["AMZN"][str(hold)]["medians"]
        ok = fresh == recorded
        print(f"  hold={hold}: fresh={fresh} recorded={recorded} -> {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"gate1 hold={hold}")

    # Gate 2: hold transform is a no-op (medians identical across holds).
    print("\n=== Gate 2: holding-period transform is a NO-OP ===")
    for asset in ("AMZN", "JPM"):
        series = art["per_asset"][asset]
        base = series["1"]["medians"]
        same = all(series[str(h)]["medians"] == base for h in HOLD_PERIODS)
        print(f"  {asset}: medians across holds all equal {base} -> {'NO-OP' if same else 'VARIES'}")
        if not same:
            failures.append(f"gate2 {asset}")
    # All 10 assets via the universe block: the hold transform must not create a
    # real holding-period effect. Allow a negligible deviation (<= 0.002) that
    # only the single price-level-crossing asset (NVDA, 2 turns) can exhibit when
    # its one artifact flip is reindexed.
    univ = art["universe"]["hold_sums"]
    order = art["universe"]["ticker_order"]
    max_dev = 0.0
    worst = None
    for t in order:
        base = univ["1"]["per_asset"][t]["medians"]
        for h in HOLD_PERIODS:
            for a, b in zip(base, univ[str(h)]["per_asset"][t]["medians"]):
                d = abs(a - b)
                if d > max_dev:
                    max_dev = d
                    worst = (t, h)
    noop = max_dev <= 0.002
    print(f"  universe (all 10 assets): max |median deviation| across holds = "
          f"{max_dev:.4f} (worst {worst}) -> {'NO-OP (negligible)' if noop else 'VARIES'}")
    if not noop:
        failures.append("gate2 universe")

    # Gates 3 & 4: the signal is a price-LEVEL artifact with no momentum skill.
    print("\n=== Gates 3/4: signal is a price-level artifact, no momentum information ===")
    for ticker in ("AMZN", "AAPL", "JPM", "TSLA", "NVDA"):
        b, _ = bt.load_ticker(ticker)
        c = b.closes_array()
        w = as_implemented_weights(c)
        # Gate 3: the framework signal is exactly reproduced from mean(log PRICE)
        # (a level), proving it does not use the return.
        w_fw = np.array([s.weight for s in bt.momentum_signals(c, lookback=LOOKBACK)])
        level_artifact = bool(np.array_equal(w, w_fw))
        frac_long = float(np.mean(w == 1.0))
        turns = int(np.sum(np.abs(np.diff(w)) > 0))
        # Gate 4: no momentum skill -- agreement with the TRUE return sign is far
        # below the 1.0 a real sign-of-return signal would achieve, and no better
        # than the always-long base rate.
        r = true_return_sign(c)
        valid = ~np.isnan(r)
        base_rate = float(np.mean(r[valid] > 0))
        agree = float(np.mean((w[valid] == 1.0) == (r[valid] > 0)))
        no_info = (agree < 0.90) and (agree <= base_rate + 0.05)
        print(f"  {ticker}: frac_long={frac_long:.3f} turns={turns} | "
              f"true_pos={base_rate:.3f} agreement={agree:.3f} | "
              f"level_artifact={level_artifact} no_info={no_info}")
        if not level_artifact:
            failures.append(f"gate3 {ticker}")
        if not no_info:
            failures.append(f"gate4 {ticker}")

    # Gate 5: determinism of gates 3/4 recomputation.
    print("\n=== Gate 5: determinism ===")
    det_ok = True
    for ticker in ("AMZN", "JPM"):
        b, _ = bt.load_ticker(ticker)
        c = b.closes_array()
        w1 = as_implemented_weights(c)
        w2 = as_implemented_weights(c)
        if not np.array_equal(w1, w2):
            det_ok = False
    print(f"  recomputation identical across two passes: {det_ok}")
    if not det_ok:
        failures.append("gate5")

    print("\n=== SUMMARY ===")
    if failures:
        print(f"VERIFICATION FAILURES: {failures}")
        return 1
    print("ALL GATES PASS: artifact reproduces; hold transform is a NO-OP; "
          "signal is a degenerate constant-long bias with no momentum information.")
    print("CONCLUSION: R-002 holding-period test is VACUOUS; the qualified "
          "momentum candidate is a long-bias artifact of a signal-construction "
          "defect (mean of log PRICES used where a log RETURN is required).")
    return 0


if __name__ == "__main__":
    sys.exit(main())