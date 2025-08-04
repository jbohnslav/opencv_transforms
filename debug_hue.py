import numpy as np
from PIL import Image
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F

# Load the image
pil_image = Image.open(
    "/Users/jrb/code/opencv_transforms/.conductor/cahokia/tests/fixtures/beans_dataset_image.jpg"
)
image = np.array(pil_image).copy()

# Test identity: hue_factor = 0.0 should return original
pil_identity = F_pil.adjust_hue(pil_image, 0.0)
np_identity = F.adjust_hue(image, 0.0)

# Compare
pil_array = np.array(pil_identity)
diff = pil_array.astype(int) - np_identity.astype(int)

print(f"PIL result shape: {pil_array.shape}")
print(f"OpenCV result shape: {np_identity.shape}")
print(f"Original shape: {image.shape}")
print(f"Max difference: {np.abs(diff).max()}")
print(f"Number of pixels with differences: {np.sum(np.abs(diff) > 0)}")
print(f"Total pixels: {np.prod(diff.shape[:2])}")

# Show some differences
indices = np.where(np.abs(diff).max(axis=2) > 0)
if len(indices[0]) > 0:
    print("\nFirst 10 pixel differences:")
    for i in range(min(10, len(indices[0]))):
        y, x = indices[0][i], indices[1][i]
        print(
            f"  Pixel ({y}, {x}): PIL={pil_array[y, x]}, OpenCV={np_identity[y, x]}, Original={image[y, x]}"
        )
