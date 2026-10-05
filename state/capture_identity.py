import os, json
print(json.dumps({k: os.environ.get(k) for k in [
    "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_REF_NAME"]}))
