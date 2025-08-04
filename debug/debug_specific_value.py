# For the test image, mean is approximately 119.898048
mean = 119.898048

# Test pixel value 24 with contrast 1.0
val = 24
cf = 1.0

# Our formula
result1 = (val - mean) * cf + mean
print(f"Raw result: {result1}")
print(f"int(result + 0.5): {int(result1 + 0.5)}")

# What if PIL is doing something different?
# Maybe PIL truncates instead of rounds for certain cases?
print(f"int(result): {int(result1)}")

# Or maybe PIL uses the degenerate value differently
degenerate = int(mean + 0.5)  # 120
print(f"\nDegenerate: {degenerate}")

# For contrast=1.0, blend formula gives:
# result = val * 1.0 + degenerate * 0.0 = val
print(f"Blend formula result: {val}")

# The issue: for contrast=1.0, PIL changes 24->23
# This suggests PIL might have a precision issue

# Let's check the exact calculation
print("\nExact calculation:")
print(f"24 - 119.898048 = {24 - 119.898048}")
print(f"(24 - 119.898048) * 1.0 + 119.898048 = {(24 - 119.898048) * 1.0 + 119.898048}")
print(f"Should be 24, but due to floating point: {24.0 - mean + mean}")

# Floating point precision issue
print("\nFloating point check:")
print(f"24 == 24.0 - mean + mean: {24.0 - mean + mean == 24}")
print(f"Difference: {24 - (24.0 - mean + mean)}")
