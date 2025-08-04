import cv2
import numpy as np
from datasets import load_dataset
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F

# Load first image from train set
dataset = load_dataset("beans", split="train", streaming=True)
sample = next(iter(dataset))
pil_image = sample["image"]
image = np.array(pil_image).copy()
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Convert PIL image to grayscale
pil_image = pil_image.convert("L")

# Test with contrast 0.5 and 1.0
for contrast_factor in [0.5, 1.0]:
    print(f"\n=== Contrast factor: {contrast_factor} ===")

    pil_enhanced = F_pil.adjust_contrast(pil_image, contrast_factor)
    np_enhanced = F.adjust_contrast(image, contrast_factor)

    pil_array = np.array(pil_enhanced)
    cv_squeezed = np_enhanced.squeeze()

    # Find differences
    diff_mask = pil_array != cv_squeezed
    if np.any(diff_mask):
        print("Arrays are different!")
        indices = np.where(diff_mask)
        print(f"Number of different pixels: {len(indices[0])}")

        # Sample a few differences
        for i in range(min(5, len(indices[0]))):
            r, c = indices[0][i], indices[1][i]
            print(
                f"  Pixel ({r},{c}): PIL={pil_array[r, c]}, CV={cv_squeezed[r, c]}, diff={pil_array[r, c] - cv_squeezed[r, c]}"
            )
    else:
        print("Arrays are identical!")

# Check a specific computation
print("\n=== Checking specific pixel calculation ===")
pixel_val = image[100, 100]
mean_val = cv2.mean(image)[0]
print(f"Pixel value: {pixel_val}")
print(f"Mean value: {mean_val}")

for cf in [0.5, 1.0]:
    result = (pixel_val - mean_val) * cf + mean_val
    result_rounded = int(result + 0.5) if result >= 0 else int(result - 0.5)
    print(f"Contrast {cf}: ({pixel_val} - {mean_val}) * {cf} + {mean_val} = {result}")
    print(f"  Rounded: {result_rounded}, Clipped: {np.clip(result_rounded, 0, 255)}")
