#!/usr/bin/env python3

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as pil_transforms

from opencv_transforms import transforms

# Create a simple test image
test_img = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
cv_img = np.array(test_img)

# Test rotation with 10 degrees
torch.manual_seed(42)
pil_rotated = pil_transforms.RandomRotation(10)(test_img)

torch.manual_seed(42)
cv_rotated = transforms.RandomRotation(10)(cv_img)

print("PIL result shape:", np.array(pil_rotated).shape)
print("CV result shape:", cv_rotated.shape)

# Check pixel differences
pil_array = np.array(pil_rotated).astype(np.float32)
cv_array = cv_rotated.astype(np.float32)

diff = np.abs(pil_array - cv_array)
print("Max difference:", diff.max())
print("Mean difference:", diff.mean())
print("Std difference:", diff.std())

# Check if CV output is still mostly black
unique_values = np.unique(cv_array)
print("CV unique values (first 20):", unique_values[:20])
print("Number of zero pixels in CV:", np.sum(cv_array == 0))
print("Total pixels:", cv_array.size)
print("Percentage of zero pixels:", 100 * np.sum(cv_array == 0) / cv_array.size)
