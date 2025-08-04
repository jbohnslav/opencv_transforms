import numpy as np
from PIL import Image
from PIL import ImageEnhance

# Create a simple test case
test_array = np.array([[23, 147, 140], [71, 133, 155]], dtype=np.uint8)
pil_img = Image.fromarray(test_array, mode="L")

# Calculate mean
mean = np.mean(test_array)
print(f"Mean: {mean}")

# Test contrast factors
for cf in [0.5, 1.0]:
    print(f"\n=== Contrast factor {cf} ===")

    # PIL result
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_result = enhancer.enhance(cf)
    pil_array = np.array(pil_result)
    print(f"PIL result:\n{pil_array}")

    # Manual calculation
    print("\nManual calculations:")
    for val in [23, 71, 133, 147, 155]:
        result = (val - mean) * cf + mean
        print(f"  {val}: ({val} - {mean:.6f}) * {cf} + {mean:.6f} = {result:.6f}")
        print(f"       int(result + 0.5) = {int(result + 0.5)}")

    # Our LUT approach
    table = np.array([(i - mean) * cf + mean for i in range(256)])
    print("\nTable values for test pixels:")
    for val in [23, 71, 133, 147, 155]:
        print(f"  table[{val}] = {table[val]:.6f} -> {int(table[val] + 0.5)}")
