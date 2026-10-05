import os, sys

for k in ["GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_REF_NAME"]:
    print(k, "=", os.environ.get(k))
