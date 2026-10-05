#!/usr/bin/env python3
import json
for p in ["state/activation_status.json", "research/data/manifest.json"]:
    d = json.load(open(p))
    print(p, "-> valid JSON,", len(d) if isinstance(d, dict) else len(d), "top-level entries")
print("OK")
