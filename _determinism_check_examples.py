"""Determinism harness: run each example twice and compare outputs."""
import hashlib
import io
import sys
from contextlib import redirect_stdout

sys.path.insert(0, ".")

examples = [
    ("examples.ma_crossover", "ma_crossover"),
    ("examples.volatility_regime_filter", "volatility_regime_filter"),
    ("examples.regime_stability_demo", "regime_stability_demo"),
]

for mod, name in examples:
    out_a = io.StringIO()
    out_b = io.StringIO()
    with redirect_stdout(out_a):
        exec(f"import {mod}; {mod}.main()")
    with redirect_stdout(out_b):
        exec(f"import {mod}; {mod}.main()")
    a, b = out_a.getvalue(), out_b.getvalue()
    ha = hashlib.sha256(a.encode()).hexdigest()
    hb = hashlib.sha256(b.encode()).hexdigest()
    same = ha == hb
    with open(f"/tmp/{name}_run1.txt", "w") as f:
        f.write(a)
    with open(f"/tmp/{name}_run2.txt", "w") as f:
        f.write(b)
    print(f"{name}: {'IDENTICAL' if same else 'DIFFERENT'}  {ha}")
