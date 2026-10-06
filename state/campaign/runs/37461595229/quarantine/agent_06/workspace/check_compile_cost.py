import ast, sys
for p in sys.argv[1:]:
    with open(p) as f:
        ast.parse(f.read())
    print("OK:", p)
