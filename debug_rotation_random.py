#!/usr/bin/env python3

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as pil_transforms

from opencv_transforms import functional as F
from opencv_transforms import transforms

# Use the same test image as the actual test
np.random.seed(123)
test_img = Image.fromarray(np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8))
cv_img = np.array(test_img)

print("=== Testing Random Rotation Parameter Generation ===")

# Test with the exact same setup as the failing test
degrees = 10

# First test - exact reproduction of test
torch.manual_seed(42)
random.seed(42)
pil_rotated = pil_transforms.RandomRotation(degrees)(test_img)

torch.manual_seed(42)
random.seed(42)
cv_rotated = transforms.RandomRotation(degrees)(cv_img)

# Check the angles generated
torch.manual_seed(42)
random.seed(42)
pil_transform = pil_transforms.RandomRotation(degrees)
pil_angle = None  # Can't easily extract

torch.manual_seed(42)
random.seed(42)
cv_transform = transforms.RandomRotation(degrees)
cv_angle = cv_transform.get_params(cv_transform.degrees)

print(f"CV generated angle: {cv_angle}")

# Try direct rotation with the same angle
direct_pil = test_img.rotate(cv_angle)
direct_cv = F.rotate(cv_img, cv_angle)

# Compare all results
pil_array = np.array(pil_rotated).astype(np.float32)
cv_array = cv_rotated.astype(np.float32)
direct_pil_array = np.array(direct_pil).astype(np.float32)
direct_cv_array = direct_cv.astype(np.float32)

print("\nRandomRotation comparison:")
diff_random = np.abs(pil_array - cv_array)
print(f"Max diff: {diff_random.max()}, Mean diff: {diff_random.mean()}")

print("\nDirect rotation comparison:")
diff_direct = np.abs(direct_pil_array - direct_cv_array)
print(f"Max diff: {diff_direct.max()}, Mean diff: {diff_direct.mean()}")

print("\nPIL RandomRotation vs PIL direct rotation:")
diff_pil = np.abs(pil_array - direct_pil_array)
print(f"Max diff: {diff_pil.max()}, Mean diff: {diff_pil.mean()}")

# Check if there are any non-zero pixels in the difference
nonzero_diff = np.count_nonzero(diff_random)
total_pixels = diff_random.size
print(
    f"\nPixels with differences: {nonzero_diff} / {total_pixels} ({100 * nonzero_diff / total_pixels:.2f}%)"
)

# Print some sample pixel values to see what's happening
print(f"\nSample PIL values: {pil_array[250, 250]}")
print(f"Sample CV values: {cv_array[250, 250]}")
print(f"Sample difference: {diff_random[250, 250]}")
