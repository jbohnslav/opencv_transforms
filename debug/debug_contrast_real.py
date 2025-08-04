import cv2
import numpy as np
from datasets import load_dataset
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F

# Load the actual test image
dataset = load_dataset("beans", split="test", cache_dir="tests/.cache/")
pil_image = dataset[0]["image"]
image = np.array(pil_image).copy()
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Convert PIL image to grayscale
pil_image = pil_image.convert("L")

contrast_factor = 0.5

pil_enhanced = F_pil.adjust_contrast(pil_image, contrast_factor)
np_enhanced = F.adjust_contrast(image, contrast_factor)

print("Original grayscale image shape:", image.shape)
print("PIL enhanced shape:", np.array(pil_enhanced).shape)
print("OpenCV enhanced shape:", np_enhanced.shape)

# Compare values
pil_array = np.array(pil_enhanced)
cv_squeezed = np_enhanced.squeeze()

print("\nAre they equal?", np.array_equal(pil_array, cv_squeezed))

# Find the first different pixel
diff_mask = pil_array != cv_squeezed
if np.any(diff_mask):
    indices = np.where(diff_mask)
    first_diff_idx = (indices[0][0], indices[1][0])
    print(f"\nFirst difference at pixel {first_diff_idx}:")
    print(f"PIL value: {pil_array[first_diff_idx]}")
    print(f"OpenCV value: {cv_squeezed[first_diff_idx]}")

    # Show surrounding pixels
    r, c = first_diff_idx
    print("\nPIL surrounding pixels:")
    print(pil_array[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2])
    print("\nOpenCV surrounding pixels:")
    print(cv_squeezed[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2])
else:
    print("Arrays are identical!")

# Check if it's a shape issue
print("\nDetailed shape info:")
print("np_enhanced shape:", np_enhanced.shape)
print("np_enhanced dtype:", np_enhanced.dtype)
print("pil_array dtype:", pil_array.dtype)
