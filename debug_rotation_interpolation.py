#!/usr/bin/env python3

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as pil_transforms

from opencv_transforms import functional as F
from opencv_transforms import transforms

# Create a simple test pattern
test_pattern = np.zeros((50, 50, 3), dtype=np.uint8)
test_pattern[10:40, 10:40] = [255, 0, 0]  # Red square
test_pattern[20:30, 20:30] = [0, 255, 0]  # Green square inside
test_img = Image.fromarray(test_pattern)
cv_img = test_pattern.copy()

print("=== Testing different interpolation methods ===")
angle = 10

# Test PIL default (should be NEAREST)
torch.manual_seed(42)
pil_result = pil_transforms.RandomRotation(angle)(test_img)
print("PIL result sample:", np.array(pil_result)[25, 25])

# Test our NEAREST (current default)
torch.manual_seed(42)
cv_result_nearest = transforms.RandomRotation(angle)(cv_img)
print("CV NEAREST result sample:", cv_result_nearest[25, 25])

# Test our BILINEAR
torch.manual_seed(42)
cv_result_bilinear = F.rotate(
    cv_img,
    transforms.RandomRotation.get_params((-angle, angle)),
    resample=cv2.INTER_LINEAR,
)
print("CV BILINEAR result sample:", cv_result_bilinear[25, 25])

# Compare differences
pil_array = np.array(pil_result)
diff_nearest = np.abs(
    pil_array.astype(np.float32) - cv_result_nearest.astype(np.float32)
)
diff_bilinear = np.abs(
    pil_array.astype(np.float32) - cv_result_bilinear.astype(np.float32)
)

print(
    "\nNearest interpolation - Max diff:",
    diff_nearest.max(),
    "Mean diff:",
    diff_nearest.mean(),
)
print(
    "Bilinear interpolation - Max diff:",
    diff_bilinear.max(),
    "Mean diff:",
    diff_bilinear.mean(),
)

# Check what PIL actually uses by checking a simple rotation
simple_pil = test_img.rotate(angle)
simple_cv_nearest = F.rotate(cv_img, angle, resample=cv2.INTER_NEAREST)
simple_cv_bilinear = F.rotate(cv_img, angle, resample=cv2.INTER_LINEAR)

simple_pil_array = np.array(simple_pil)
simple_diff_nearest = np.abs(
    simple_pil_array.astype(np.float32) - simple_cv_nearest.astype(np.float32)
)
simple_diff_bilinear = np.abs(
    simple_pil_array.astype(np.float32) - simple_cv_bilinear.astype(np.float32)
)

print("\nSimple rotation comparison:")
print(
    "PIL vs CV NEAREST - Max diff:",
    simple_diff_nearest.max(),
    "Mean diff:",
    simple_diff_nearest.mean(),
)
print(
    "PIL vs CV BILINEAR - Max diff:",
    simple_diff_bilinear.max(),
    "Mean diff:",
    simple_diff_bilinear.mean(),
)
