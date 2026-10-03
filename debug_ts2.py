import numpy as np
ts = [1230906600, 1231165800, 1231252200]
timestamps = np.asarray(ts, dtype=np.int64)
print("dtype:", timestamps.dtype)
for t in timestamps:
    print(type(t), t // 86400)
