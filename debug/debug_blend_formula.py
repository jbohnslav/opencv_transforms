import numpy as np
from PIL import Image
from PIL import ImageEnhance

# Create a simple test with known values
test_array = np.array([[39, 79, 120, 168]], dtype=np.uint8)
pil_img = Image.fromarray(test_array, mode="L")

# Get mean
mean = np.mean(test_array)
print(f"Mean: {mean}")
print(f"Degenerate value (int(mean + 0.5)): {int(mean + 0.5)}")

# Test different formulas
degenerate = int(mean + 0.5)

for cf in [0.5, 1.0]:
    print(f"\n=== Contrast factor {cf} ===")

    # PIL result
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_result = np.array(enhancer.enhance(cf))
    print(f"PIL result: {pil_result[0]}")

    # Formula 1: blend = image * factor + degenerate * (1 - factor)
    formula1 = np.array(
        [int(val * cf + degenerate * (1 - cf) + 0.5) for val in test_array[0]]
    )
    print(f"Formula 1: {formula1}")

    # Formula 2: the original contrast formula
    formula2 = np.array([int((val - mean) * cf + mean + 0.5) for val in test_array[0]])
    print(f"Formula 2: {formula2}")

    # Check which matches
    print(f"Formula 1 matches PIL: {np.array_equal(formula1, pil_result[0])}")
    print(f"Formula 2 matches PIL: {np.array_equal(formula2, pil_result[0])}")
