import numpy as np
a = np.arange(2500.0)
print("a[-10.0:30].size:", a[-10.0:30].size)
print("a[29-40+1:30].size:", a[29-40+1:30].size)
print("a[int(29)-int(40.0)+1:int(30)].size:", a[int(29)-int(40.0)+1:int(30)].size)
