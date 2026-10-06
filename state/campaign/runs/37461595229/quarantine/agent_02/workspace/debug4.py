path = "state/LEARNING_STATE.md"
lines = open(path).read().split("\n")
print("total lines:", len(lines))
print("--- tail ---")
print("".join(lines[-3:]))
print("--- lines 11-13 ---")
for i in (10, 11, 12):
    if i < len(lines):
        print(f"[{i+1}] len={len(lines[i])}")
print("--- frontier rows ---")
for i in (29, 30, 31, 32):
    if i < len(lines):
        print(f"[{i+1}] len={len(lines[i])} tail={lines[i][-80:]!r}")
print("--- search for cost row ---")
for j, l in enumerate(lines, 1):
    if "cost-robustness" in l or "COST_ROBUST=6" in l:
        print(f"FOUND at line {j}: {l[:120]}")
print("--- search for momentum op ---")
for j, l in enumerate(lines, 1):
    if "single-point lookback" in l:
        print(f"FOUND at line {j}: tail={l[-100:]!r}")
