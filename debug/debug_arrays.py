import cv2
import numpy as np
from datasets import load_dataset
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F

# Load the test image
dataset = load_dataset("beans", split="test", cache_dir="tests/.cache/")
pil_image = dataset[0]["image"]
image = np.array(pil_image).copy()
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Convert PIL image to grayscale
pil_image = pil_image.convert("L")

# Test with contrast factor 1.0 (should be identity)
contrast_factor = 1.0

pil_enhanced = F_pil.adjust_contrast(pil_image, contrast_factor)
np_enhanced = F.adjust_contrast(image, contrast_factor)

pil_array = np.array(pil_enhanced)
cv_squeezed = np_enhanced.squeeze()

print("Shapes:")
print(f"  PIL: {pil_array.shape}")
print(f"  CV:  {cv_squeezed.shape}")

print("\nAre arrays equal?", np.array_equal(pil_array, cv_squeezed))

# Check if arrays are close (allowing small differences)
print("Are arrays close (atol=1)?", np.allclose(pil_array, cv_squeezed, atol=1))

# Check for differences
diff = pil_array != cv_squeezed
num_diff = np.sum(diff)
print(f"\nNumber of different pixels: {num_diff}")

if num_diff > 0 and num_diff < 50:
    # Show the different pixels
    indices = np.where(diff)
    print("\nDifferent pixels (first 10):")
    for i in range(min(10, len(indices[0]))):
        r, c = indices[0][i], indices[1][i]
        print(
            f"  ({r},{c}): PIL={pil_array[r, c]}, CV={cv_squeezed[r, c]}, original={image[r, c]}"
        )

# Check if identity transform works
print("\nFor contrast=1.0, should be identity transform")
print(f"Original == PIL result: {np.array_equal(image, pil_array)}")
print(f"Original == CV result:  {np.array_equal(image, cv_squeezed)}")
