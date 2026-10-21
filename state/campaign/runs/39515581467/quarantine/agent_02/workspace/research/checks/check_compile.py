import py_compile
py_compile.compile("research/checks/momentum_cost_robustness.py", doraise=True)
py_compile.compile("research/checks/verify_momentum_cost_robustness.py", doraise=True)
py_compile.compile("research/checks/smoke_momentum_cost_robustness.py", doraise=True)
print("COMPILE OK")