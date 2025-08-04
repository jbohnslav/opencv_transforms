import cv2
import numpy as np
from datasets import load_dataset
from torchvision.transforms import functional as F_pil

# Load the actual test image
dataset = load_dataset("beans", split="test", cache_dir="tests/.cache/")
pil_image = dataset[0]["image"]
image = np.array(pil_image).copy()
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Convert PIL image to grayscale
pil_image = pil_image.convert("L")

# Focus on the pixel that differs
row, col = 278, 2
pixel_value = image[row, col]
print(f"Original pixel value at ({row},{col}): {pixel_value}")

# Calculate mean
mean_value = cv2.mean(image)[0]
mean_value_rounded = round(mean_value)
print("\nMean calculation:")
print(f"cv2.mean: {mean_value}")
print(f"rounded: {mean_value_rounded}")

# Manually compute contrast adjustment with factor 0.5
contrast_factor = 0.5
manual_result = (
    pixel_value - mean_value_rounded
) * contrast_factor + mean_value_rounded
print("\nManual calculation:")
print(
    f"({pixel_value} - {mean_value_rounded}) * {contrast_factor} + {mean_value_rounded} = {manual_result}"
)
print(f"Clipped to uint8: {int(np.clip(manual_result, 0, 255))}")

# Check what PIL does
pil_enhanced = F_pil.adjust_contrast(pil_image, contrast_factor)
pil_result = np.array(pil_enhanced)[row, col]
print(f"\nPIL result: {pil_result}")

# Check exact mean value used by PIL
# PIL uses the mean of the entire image
pil_array = np.array(pil_image)
pil_mean = np.mean(pil_array)
print("\nPIL mean calculation:")
print(f"np.mean: {pil_mean}")

# Compute with PIL's mean
pil_manual = (pixel_value - pil_mean) * contrast_factor + pil_mean
print("\nPIL-style calculation:")
print(f"({pixel_value} - {pil_mean}) * {contrast_factor} + {pil_mean} = {pil_manual}")
print(f"As int: {int(pil_manual)}")

# Let's check the LUT table value
table = (
    np.array(
        [
            (i - mean_value_rounded) * contrast_factor + mean_value_rounded
            for i in range(256)
        ]
    )
    .clip(0, 255)
    .astype("uint8")
)
print(f"\nLUT table value at {pixel_value}: {table[pixel_value]}")
