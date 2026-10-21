#!/usr/bin/env python3
# Test comprehension syntax to understand the issue

print("Testing comprehension syntax...")

# This should work
result1 = {
    str(h): 'value'
    for h in (1, 2, 3, 5)
}
print("Simple comprehension works:", result1)

# This should also work
result2 = {
    str(h): (
        'value'
        if True
        else 'other'
    )
    for h in (1, 2, 3, 5)
}
print("Conditional comprehension works:", result2)

# This is more complex but should work
result3 = {
    str(h): (
        'A' if all([True, True])
        else 'B'
    )
    for h in (1, 2, 3, 5)
}
print("Complex comprehension works:", result3)

print("All tests passed!")