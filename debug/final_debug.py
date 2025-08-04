import cv2
import numpy as np
from datasets import load_dataset
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F

# Load the test setup
train_dataset = load_dataset(
    "beans", split="train", streaming=True, cache_dir="tests/.cache/"
)
samples = []
for i, sample in enumerate(train_dataset):
    if i >= 50:
        break
    samples.append(sample["image"])

pil_image = samples[0]
image = np.array(pil_image).copy()
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
pil_gray = pil_image.convert("L")

# Test all contrast factors
for contrast_factor in [0.5, 1.0]:
    print(f"\n=== Contrast factor {contrast_factor} ===")

    pil_enhanced = F_pil.adjust_contrast(pil_gray, contrast_factor)
    np_enhanced = F.adjust_contrast(image_gray, contrast_factor)

    pil_array = np.array(pil_enhanced)
    cv_squeezed = np_enhanced.squeeze()

    print(f"Shapes: PIL={pil_array.shape}, CV={cv_squeezed.shape}")
    print(f"Are equal? {np.array_equal(pil_array, cv_squeezed)}")

    # Find differences
    diff_mask = pil_array != cv_squeezed
    num_diff = np.sum(diff_mask)
    print(f"Number of different pixels: {num_diff}")

    if num_diff > 0 and num_diff < 20:
        indices = np.where(diff_mask)
        for i in range(min(5, len(indices[0]))):
            r, c = indices[0][i], indices[1][i]
            print(
                f"  ({r},{c}): PIL={pil_array[r, c]}, CV={cv_squeezed[r, c]}, diff={int(pil_array[r, c]) - int(cv_squeezed[r, c])}"
            )
