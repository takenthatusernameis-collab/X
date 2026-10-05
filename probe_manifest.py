import json, os
p = os.path.join(os.getcwd(), "research", "data", "manifest.json")
m = json.load(open(p))
print("dataset_id:", m["dataset_id"])
print("collection:", m.get("collection_date"))
print("universe count:", len(m.get("universe", [])))
for e in m["entries"][:3]:
    print(" ", e["ticker"], e.get("location"))
print("total entries:", len(m["entries"]))
