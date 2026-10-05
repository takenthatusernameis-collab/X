#!/usr/bin/env python3
"""Helper: print GitHub Actions identity/env vars + current UTC time.
Scratch/diagnostic helper permitted to persist by PERSISTENCE_POLICY.md."""
import os, datetime
for k in ["GITHUB_RUN_ID","GITHUB_RUN_ATTEMPT","GITHUB_SHA","GITHUB_REF_NAME","GITHUB_REPOSITORY"]:
    print(f"{k}={os.environ.get(k,'(not set)')}")
print("UTC NOW:", datetime.datetime.utcnow().isoformat() + "Z")
