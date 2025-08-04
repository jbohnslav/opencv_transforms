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

contrast_factor = 0.0

pil_enhanced = F_pil.adjust_contrast(pil_image, contrast_factor)
np_enhanced = F.adjust_contrast(image, contrast_factor)

print("Original grayscale image shape:", image.shape)
print("PIL enhanced shape:", np.array(pil_enhanced).shape)
print("OpenCV enhanced shape:", np_enhanced.shape)

# Compare values
pil_array = np.array(pil_enhanced)
cv_squeezed = np_enhanced.squeeze()

# Check unique values - should be all the mean value
print("\nUnique values in PIL result:", np.unique(pil_array))
print("Unique values in OpenCV result:", np.unique(cv_squeezed))

# Check mean calculation
mean_value = cv2.mean(image)[0]
print(f"\nMean value (float): {mean_value}")
print(f"Mean value (int): {int(mean_value)}")
print(f"Mean value (rounded): {round(mean_value)}")

# Check what PIL does
pil_mean = np.mean(np.array(pil_image))
print(f"\nPIL mean (float): {pil_mean}")
print(f"PIL mean (int): {int(pil_mean)}")
print(f"PIL mean (rounded): {round(pil_mean)}")

# Manual calculation
print("\nFor contrast_factor=0:")
print("Result should be: (pixel - mean) * 0 + mean = mean")
print(f"OpenCV uses mean={mean_value}, results in {int(mean_value)}")
print(f"PIL appears to use rounded mean={round(pil_mean)}")
