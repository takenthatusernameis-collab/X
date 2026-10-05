import json
p = "/home/runner/work/X/X/state/activation_status.json"
a = json.load(open(p))
print("JSON OK")
print("keys:", sorted(a.keys()))
print("activation_id:", a["activation_id"], "| status:", a["status"], "| acceptance:", a["acceptance"][:60], "...")
