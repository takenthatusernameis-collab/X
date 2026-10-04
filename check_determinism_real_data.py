"""Run the real-data MA-crossover example and report its sha256, for determinism checks."""
from __future__ import annotations

import hashlib
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-B", "-m", "examples.ma_crossover_real_data"],
    capture_output=True,
    text=True,
    cwd="/home/runner/work/X/X",
)
text = result.stdout + result.stderr
print("sha256:", hashlib.sha256(text.encode("utf-8")).hexdigest())
print("rc:", result.returncode)
if result.returncode != 0:
    print(result.stdout[-4000:])
    print(result.stderr[-4000:])
    sys.exit(result.returncode)
