from pathlib import Path
lines = Path("state/STATE.md").read_text().split("\n")
for i in range(1505, 1524):
    print(i + 1, repr(lines[i]))