# Load a specific seed to match the test
import random

import cv2
import numpy as np
from datasets import load_dataset
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F

random.seed(1)  # Same as the test uses

# Load the actual test images
dataset = load_dataset("beans", split="test", cache_dir="tests/.cache/")

# Get all test images
pil_images = [dataset[i]["image"] for i in range(len(dataset))]

# Select random image (same way the test does)
idx = random.randint(0, len(pil_images) - 1)
print(f"Selected image index: {idx}")

pil_image = pil_images[idx]
image = np.array(pil_image).copy()
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Convert PIL image to grayscale
pil_image = pil_image.convert("L")

contrast_factor = 0.0

pil_enhanced = F_pil.adjust_contrast(pil_image, contrast_factor)
np_enhanced = F.adjust_contrast(image, contrast_factor)

# Compare values
pil_array = np.array(pil_enhanced)
cv_squeezed = np_enhanced.squeeze()

# Check unique values - should be all the mean value
print("\nUnique values in PIL result:", np.unique(pil_array))
print("Unique values in OpenCV result:", np.unique(cv_squeezed))

# Check mean calculation
mean_value = cv2.mean(image)[0]
print(f"\nMean value (float): {mean_value}")

# Check what PIL does
pil_mean = np.mean(np.array(pil_image))
print(f"PIL mean (float): {pil_mean}")

# Show the actual difference
print(f"\nPIL result unique value: {pil_array[0, 0]}")
print(f"OpenCV result unique value: {cv_squeezed[0, 0]}")
