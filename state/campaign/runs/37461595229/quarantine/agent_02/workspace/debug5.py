path = "state/LEARNING_STATE.md"
lines = open(path).read().splitlines()
print("lines via splitlines:", len(lines))
print("lines via split:", len(open(path).read().split("\n")))
for i in (29, 30, 31, 32):
    if i < len(lines):
        l = lines[i]
        print(f"[{i+1}] len={len(l)} head={l[:70]!r} ... tail={l[-70:]!r}")
print("--- last 5 via splitlines ---")
for l in lines[-5:]:
    print(repr(l[:150]))
