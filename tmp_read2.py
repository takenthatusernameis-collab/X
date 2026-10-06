s = open("/home/runner/work/X/X/state/LEARNING_STATE.md").read()
i = s.find("Next: momentum is admitted")
print(repr(s[i-200:i+500]))
