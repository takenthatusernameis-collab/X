import json
with open("state/activation_status.json") as f:
    d = json.load(f)
required = ["activation_id", "run_attempt", "sha", "ref_name", "repository", "status", "phase",
            "objective", "start_time", "finish_time", "changed", "verified", "unverified", "next",
            "acceptance", "research_conclusion"]
missing = [k for k in required if k not in d]
print("activation_id:", d["activation_id"])
print("status:", d["status"])
print("phase:", d["phase"])
print("verdict:", d.get("verdict"))
print("missing keys:", missing if missing else "none")
print("changed count:", len(d["changed"]))
print("verified count:", len(d["verified"]))
print("JSON VALID")
