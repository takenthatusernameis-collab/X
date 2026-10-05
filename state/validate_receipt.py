import json
try:
    d = json.load(open("state/activation_status.json"))
    for k in ["activation_id", "status", "objective", "phase", "changed",
              "verified", "unverified", "next"]:
        assert k in d, f"missing key {k}"
    print("JSON RECEIPT VALID: keys present, activation_id =", d["activation_id"])
except Exception as e:
    print("INVALID:", e)
    import sys; sys.exit(1)
