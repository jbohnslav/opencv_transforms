import cv2
import numpy as np
from datasets import load_dataset
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F

# Mimic exactly what the test does
# First, load dataset for "test" split
dataset = load_dataset("beans", split="test", cache_dir="tests/.cache/")

# The test fixture loads first 50 from train split for test_images
train_dataset = load_dataset(
    "beans", split="train", streaming=True, cache_dir="tests/.cache/"
)

samples = []
for i, sample in enumerate(train_dataset):
    if i >= 50:  # Limit to 50 samples for tests
        break
    samples.append(sample["image"])

# single_test_image returns the first one
pil_image = samples[0]
image = np.array(pil_image).copy()

# Now do exactly what the test does
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
pil_gray = pil_image.convert("L")

# Test contrast=1.0
contrast_factor = 1.0
pil_enhanced = F_pil.adjust_contrast(pil_gray, contrast_factor)
np_enhanced = F.adjust_contrast(image_gray, contrast_factor)

pil_array = np.array(pil_enhanced)
cv_squeezed = np_enhanced.squeeze()

print("Are they equal?", np.array_equal(pil_array, cv_squeezed))

# Find differences
diff = pil_array != cv_squeezed
if np.any(diff):
    indices = np.where(diff)
    print(f"\nNumber of different pixels: {len(indices[0])}")
    for i in range(min(5, len(indices[0]))):
        r, c = indices[0][i], indices[1][i]
        print(
            f"  ({r},{c}): PIL={pil_array[r, c]}, CV={cv_squeezed[r, c]}, original={image_gray[r, c]}"
        )

# Check the mean calculation
mean_cv = cv2.mean(image_gray)[0]
mean_np = np.mean(image_gray)
print("\nMean calculations:")
print(f"  cv2.mean: {mean_cv}")
print(f"  np.mean:  {mean_np}")
print(f"  difference: {abs(mean_cv - mean_np)}")
