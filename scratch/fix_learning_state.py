import re
path = "/home/runner/work/X/X/state/LEARNING_STATE.md"
lines = open(path).read().split("\n")
assert lines[11].startswith("- **Decision: RETAIN (verified)** —"), lines[11][:60]
prefix = ("- **Decision: RETAIN (verified)** — the frontier-first selection produced a clean ordering "
          "(MA crossover -> reversal -> CSRS -> volatility targeting -> momentum), each cell closed with a "
          "decisive verdict and no equivalent re-tests; three cells closed FALSIFIED (negative evidence, as "
          "expected) but the momentum cell closed SUPPORTED with a genuine positive edge that survived the "
          "regime gate, the independent-verification gate, and a post-hoc engine-asymmetry investigation "
          "(the apparent reversal/momentum magnitude asymmetry was shown to be an engine warmup+train "
          "compounding artifact that understates rather than fakes the momentum edge); the current "
          "activation (37361000967) then tested the remaining lookback-robustness question on REAL data and "
          "found the edge persists across lookback 3/5/10 for 6/10 assets with 0 lookback-sensitive cases, ")
assert lines[11].startswith(prefix), "prefix mismatch"
lines[11] = (prefix +
    "this activation (current) completed the momentum-boundary review: the cross-sectional-momentum "
    "artifact (lookback 63) reproduced exactly via an independent engine walk-forward + coin-flip null "
    "path (post-repair verifier: all recomputations MATCH, including baseline null -0.013), verdict "
    "FALSIFIED (CONSISTENT_WITH_NOISE; 9/10 sub-universes CONSISTENT_WITH_NOISE, 1/10 REGIME_DEPENDENT "
    "(JNJ, dispersion > 2x null), 1/10 REGIME_STABLE (XOM, small positive stable edge); full-universe "
    "spread median +0.002 within null tol +0.0305, concentration NO_EDGE, perturbation sweep "
    "CONSISTENT_WITH_NOISE). The frontier-first delta remains RETAIN (verified): both momentum boundary "
    "cells were tested once, decisively, and independently, with no equivalent re-tests.")
open(path, "w").write("\n".join(lines))
print("line 12 rewritten; line count:", len(lines))
print("last char of line 12:", repr(lines[11][-80:]))
