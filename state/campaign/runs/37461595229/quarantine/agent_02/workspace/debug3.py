import sys
path = "state/LEARNING_STATE.md"
lines = open(path).read().split("\n")
print("total lines:", len(lines))
l32 = lines[31]
print("line 32 length:", len(l32))
print("last 100 chars:", repr(l32[-100:]))
