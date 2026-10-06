lines = open("/home/runner/work/X/X/state/LEARNING_STATE.md").readlines()
for idx in range(6, 14):
    print(idx + 1, repr(lines[idx]))
