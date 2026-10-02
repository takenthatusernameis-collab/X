with open("/home/runner/work/X/X/state/STATE.md", "r") as f:
    lines = f.readlines()
with open("/home/runner/work/X/X/_verify_section.txt", "r") as f:
    section = f.readlines()
out = []
for i, line in enumerate(lines, start=1):
    if line == "## Evidence standard\n":
        out.extend(section)
    out.append(line)
with open("/home/runner/work/X/X/state/STATE.md", "w") as f:
    f.writelines(out)
print("inserted verification section before Evidence standard")
