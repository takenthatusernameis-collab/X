"""Determinism check for the universe-wide regime-stability example.

Asserts `examples/regime_stability_universe.py` produces byte-identical
output across two independent runs; records the sha256 of the output for the
activation record. Research/simulation only.
"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "examples" / "regime_stability_universe.py"


def capture_output(script: Path) -> bytes:
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        check=True,
        env={**__import__("os").environ, "PYTHONHASHSEED": "42"},
    ).stdout


if __name__ == "__main__":
    import hashlib

    r1 = capture_output(SCRIPT)
    r2 = capture_output(SCRIPT)
    print(f"run1 sha256: {hashlib.sha256(r1).hexdigest()}")
    print(f"run2 sha256: {hashlib.sha256(r2).hexdigest()}")
    print(f"r1 == r2: {r1 == r2}")
    assert r1 == r2, "example output is not deterministic across runs"
    print("byte-identical across runs")
    sys.exit(0)
