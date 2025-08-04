import numpy as np
from PIL import Image
from PIL import ImageEnhance

# Test specific values that are problematic
test_values = [23, 24, 61, 62, 68, 69]
test_array = np.array([test_values], dtype=np.uint8)
pil_img = Image.fromarray(test_array, mode="L")

# Calculate mean
mean = np.mean(test_array)
print(f"Mean: {mean}")

# Apply contrast=1.0 (should be identity)
enhancer = ImageEnhance.Contrast(pil_img)
pil_result = enhancer.enhance(1.0)
pil_array = np.array(pil_result)

print(f"\nOriginal: {test_array[0]}")
print(f"PIL result: {pil_array[0]}")
print(f"Changed: {test_array[0] != pil_array[0]}")

# Check what PIL does internally
# PIL seems to have precision issues when mean is not an integer
print("\nChecking PIL's internal calculation:")
for val in test_values:
    # PIL calculation: val * contrast + mean * (1 - contrast)
    # For contrast=1.0: val * 1 + mean * 0 = val
    # But PIL might be doing something else...
    result = val
    print(f"  {val} -> {result} (should stay {val})")
