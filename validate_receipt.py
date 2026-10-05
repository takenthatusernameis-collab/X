import json
d = json.load(open('state/activation_status.json'))
print("keys:", sorted(d.keys()))
print("status:", d["status"])
for k in ["activation_id","objective","phase","changed","verified","unverified","next","acceptance","research_conclusion"]:
    assert k in d, k
print("structure OK")
print("num verified checks:", len(d["verified"]))
print("num changed files:", len(d["changed"]))
print("next action:", d["next"][:80])
