import numpy as np
print("numpy", np.__version__)
a = np.random.default_rng(42)
print("a standard_normal:", a.standard_normal())
b = np.random.default_rng(42)
print("b standard_normal:", b.standard_normal())
print("equal:", a.standard_normal() == b.standard_normal())
