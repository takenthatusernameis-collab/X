lines = open("/home/runner/work/X/X/state/LEARNING_STATE.md").read().split("\n")
print("line 11 (Observed Effect) length:", len(lines[10]))
print("line 11 tail:", repr(lines[10][-300:]))
print("line count:", len(lines))
print("--- frontier row check ---")
for i, l in enumerate(lines):
    if "Cross-sectional momentum" in l:
        print(f"line {i+1}: {l[:120]}...")
