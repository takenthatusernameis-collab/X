"""Determinism check for examples/regime_stability_nvda.py."""
import io
import sys
from contextlib import redirect_stdout

import examples.regime_stability_nvda as m

out1 = io.StringIO()
out2 = io.StringIO()
with redirect_stdout(out1):
    m.main()
with redirect_stdout(out2):
    m.main()
s1, s2 = out1.getvalue(), out2.getvalue()
print("r1 == r2:", s1 == s2)
if s1 != s2:
    # show first differing line
    l1, l2 = s1.splitlines(), s2.splitlines()
    for a, b in zip(l1, l2):
        if a != b:
            print("FIRST DIFFERENCE:"); print("  r1:", a); print("  r2:", b); break
else:
    print("byte-identical across runs")
