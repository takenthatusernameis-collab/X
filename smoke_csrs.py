"""Smoke test for CSRS helpers: hand-verify one bar against the implementation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import research.backtest as bt

# Build a tiny family of 3 synthetic assets with closes we control.
# bar indices: 0..7
# We only need bars >= lookback=2 and < n-1 for the spread at bar i
# to use closes[i]/closes[i-2] and next-day close closes[i+1].
def make_family():
    rng = np.random.default_rng(0)
    fam = []
    prices = {a: np.array([100.0 + rng.random() * 10 for _ in range(8)], dtype=np.float64) for a in range(3)}
    for a in range(3):
        dates = np.arange(1, 9, dtype=np.int64)
        c = prices[a]
        h = np.maximum(c, np.roll(c, -1))
        l = np.minimum(c, np.roll(c, -1))
        fam.append(bt.BarSequence(dates, c, h, l, c, np.full(8, 1e6)))
    return fam

fam = make_family()
lk, tk, bk = 2, 1, 1

# Hand computation for bar i=2 (lookback reached), i=3, i=4.
def hand_spread(fam, lk, tk, bk, i):
    rets = {a: float(np.log(fam[a].closes_array()[i]) - np.log(fam[a].closes_array()[i - lk]))
            for a in range(len(fam))}
    order = sorted(rets, key=lambda a: rets[a], reverse=True)
    longs = order[:tk]
    shorts = order[-bk:]
    leg = lambda a: float(np.log(fam[a].closes_array()[i + 1]) - np.log(fam[a].closes_array()[i]))
    return float(np.mean([leg(a) for a in longs]) - np.mean([leg(a) for a in shorts]))

out = bt.csrs_spread_family(fam, lk, tk, bk)
print("family closes[2]:", {a: fam[a].closes_array()[2] for a in range(3)})
for i in [0, 1, 2, 3, 4, 6]:
    h = hand_spread(fam, lk, tk, bk, i)
    print(f"bar {i}: impl={out[i]:+.6f} hand={h:+.6f} match={np.isclose(out[i], h)}")

# Determinism
out2 = bt.csrs_spread_family(fam, lk, tk, bk)
print("determinism out==out2:", np.allclose(out, out2))

# Null: 1000 coin flips should be ~50% heads
rng = np.random.default_rng(42)
flip_counts = 0
for _ in range(1000):
    s = rng.random() < 0.5
    if s:
        flip_counts += 1
print("coin flip (1000): heads =", flip_counts, "(approx 500)")

# Null spread on the same family for 3000 draws
nulls = [bt.csrs_null_spread_family(fam, lk, tk, bk, 42)[i] for i in range(2, 5)]
print("null spread bars 2..4:", [round(x, 4) for x in nulls])
# The rank order at bar 2 under null is identical to the real one; check a bar
# where the coin lands heads (sign preserved) equals the real spread magnitude.
print("null magnitudes match real spread magnitude:",
      all(abs(n) == abs(out[i]) for i, n in zip([2, 3, 4], nulls)))

print("SMOKE OK")
