"""Count unit tests per module."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

loader = unittest.TestLoader()
suite = loader.discover(start_dir="tests", pattern="test_*.py")

counts = {}
def walk(s):
    for test in s:
        if isinstance(test, unittest.TestSuite):
            walk(test)
        else:
            mod = test.__class__.__module__
            counts[mod] = counts.get(mod, 0) + 1

walk(suite)
total = sum(counts.values())
for mod, n in sorted(counts.items()):
    print(f"{n:3d}  {mod}")
print(f"{total:3d} total")
