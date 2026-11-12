from pathlib import Path
text = Path("state/STATE.md").read_text()
lines = text.split("\n")
print("total lines:", len(lines))
for i in range(len(lines) - 12, len(lines)):
    print(i + 1, repr(lines[i]))