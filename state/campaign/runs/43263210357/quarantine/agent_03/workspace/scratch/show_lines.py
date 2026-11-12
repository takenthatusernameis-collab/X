from pathlib import Path
lines = Path("state/LEARNING_STATE.md").read_text().split("\n")
for i in range(91, 98):
    print(i + 1, repr(lines[i][-80:]))