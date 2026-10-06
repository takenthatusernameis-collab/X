#!/usr/bin/env python
import subprocess, sys
r = subprocess.run(
    [sys.executable, "research/checks/diag_null_seed.py"],
    capture_output=True, text=True)
sys.stdout.write(r.stdout)
if r.stderr:
    sys.stderr.write(r.stderr)
sys.stdout.write("EXIT_CODE: %d\n" % r.returncode)
sys.exit(r.returncode)
