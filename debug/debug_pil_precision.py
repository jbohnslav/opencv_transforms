import numpy as np
from PIL import Image
from PIL import ImageEnhance

# Create test case with the problematic values
# Values where PIL changes them for contrast=1.0: 24->23, 148->147, 141->140
test_vals = [23, 24, 140, 141, 147, 148]
rows = []
for val in test_vals:
    rows.append([val] * 10)  # Repeat to get a proper image
test_array = np.array(rows, dtype=np.uint8)

pil_img = Image.fromarray(test_array, mode="L")
mean = np.mean(test_array)
print(f"Test array mean: {mean}")

# Apply contrast=1.0
enhancer = ImageEnhance.Contrast(pil_img)
pil_result = enhancer.enhance(1.0)
result_array = np.array(pil_result)

print("\nResults for contrast=1.0:")
for i, val in enumerate(test_vals):
    result = result_array[i, 0]
    if val != result:
        print(f"  {val} -> {result} (CHANGED!)")
    else:
        print(f"  {val} -> {result} (same)")

# Let's check PIL's degenerate image (the gray image it blends with)
# PIL creates this by converting to "L" mode and using the mean
degenerate = Image.new("L", pil_img.size, int(mean + 0.5))
deg_array = np.array(degenerate)
print(f"\nPIL's degenerate image value: {deg_array[0, 0]}")
print(f"Mean rounded: {int(mean + 0.5)}")

# PIL blends: result = image * factor + degenerate * (1 - factor)
# For factor=1.0: result = image * 1 + degenerate * 0 = image
# So it should be identity, but PIL might have precision issues
