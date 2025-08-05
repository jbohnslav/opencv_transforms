#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image

from opencv_transforms import functional as F

# Create a simpler test case to isolate the issue
test_pattern = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
test_img = Image.fromarray(test_pattern)
cv_img = test_pattern.copy()

angle = 7.645385265350342  # Same angle from debug

print("=== Testing PIL rotation settings ===")

# Test PIL rotation with different parameters
pil_result_default = test_img.rotate(angle)
pil_result_nearest = test_img.rotate(angle, resample=Image.NEAREST)
pil_result_bilinear = test_img.rotate(angle, resample=Image.BILINEAR)

# Test OpenCV with different parameters
cv_result_nearest = F.rotate(cv_img, angle, resample=cv2.INTER_NEAREST)
cv_result_bilinear = F.rotate(cv_img, angle, resample=cv2.INTER_LINEAR)

print("PIL default result sample:", np.array(pil_result_default)[10, 10])
print("PIL NEAREST result sample:", np.array(pil_result_nearest)[10, 10])
print("PIL BILINEAR result sample:", np.array(pil_result_bilinear)[10, 10])
print("CV NEAREST result sample:", cv_result_nearest[10, 10])
print("CV BILINEAR result sample:", cv_result_bilinear[10, 10])

# Compare with original image pixel values around center
print("\nOriginal image around center:")
print("Original [9,9]:", test_pattern[9, 9])
print("Original [10,10]:", test_pattern[10, 10])
print("Original [11,11]:", test_pattern[11, 11])

# Check border handling
print("\n=== Border handling test ===")
# Create an image with known border values
border_img = np.full((20, 20, 3), 128, dtype=np.uint8)  # Gray background
border_img[5:15, 5:15] = [255, 0, 0]  # Red center
border_pil = Image.fromarray(border_img)

pil_border_rotated = border_pil.rotate(angle)
cv_border_rotated = F.rotate(border_img, angle)

# Check corner pixels (should be filled with background)
print("PIL corner [0,0]:", np.array(pil_border_rotated)[0, 0])
print("CV corner [0,0]:", cv_border_rotated[0, 0])
print("PIL corner [19,19]:", np.array(pil_border_rotated)[19, 19])
print("CV corner [19,19]:", cv_border_rotated[19, 19])

# Check center rotation
print("PIL center [10,10]:", np.array(pil_border_rotated)[10, 10])
print("CV center [10,10]:", cv_border_rotated[10, 10])

border_diff = np.abs(
    np.array(pil_border_rotated).astype(np.float32)
    - cv_border_rotated.astype(np.float32)
)
print(f"Border test - Max diff: {border_diff.max()}, Mean diff: {border_diff.mean()}")
print(f"Non-zero differences: {np.count_nonzero(border_diff)}")
