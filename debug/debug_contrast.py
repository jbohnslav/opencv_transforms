import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F

# Create a simple test case
test_img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
print("Original grayscale image shape:", test_img.shape)
print("First few pixels:", test_img[:3, :3])

# Convert to PIL Image
pil_img = Image.fromarray(test_img, mode="L")

# Test with contrast factor 0.5
contrast_factor = 0.5

# Apply PIL adjustment
pil_enhanced = F_pil.adjust_contrast(pil_img, contrast_factor)
pil_array = np.array(pil_enhanced)

# Apply OpenCV adjustment
cv_enhanced = F.adjust_contrast(test_img, contrast_factor)

print("\nPIL enhanced shape:", pil_array.shape)
print("OpenCV enhanced shape:", cv_enhanced.shape)

print("\nPIL first few pixels:", pil_array[:3, :3])
print("OpenCV first few pixels:", cv_enhanced[:3, :3])

# Check if they're equal
cv_enhanced_squeezed = cv_enhanced.squeeze() if cv_enhanced.ndim == 3 else cv_enhanced

print("\nOpenCV squeezed shape:", cv_enhanced_squeezed.shape)
print("Are they equal?", np.array_equal(pil_array, cv_enhanced_squeezed))

# Find differences
diff = pil_array.astype(int) - cv_enhanced_squeezed.astype(int)
print("\nDifference stats:")
print("Max diff:", np.max(np.abs(diff)))
print("Mean diff:", np.mean(np.abs(diff)))
print("Number of different pixels:", np.sum(diff != 0))

# Verify mean calculation
print("\nMean value used:")
print("Direct mean:", np.mean(test_img))
print("CV2 mean:", cv2.mean(test_img)[0])
print("Rounded CV2 mean:", round(cv2.mean(test_img)[0]))
