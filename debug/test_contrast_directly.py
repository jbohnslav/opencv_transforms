import cv2
import numpy as np
from datasets import load_dataset
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F

# Test with multiple different images and contrast factors
dataset = load_dataset("beans", split="train", streaming=True)

mismatches = []
for i, sample in enumerate(dataset):
    if i >= 10:  # Test first 10 images
        break

    pil_image = sample["image"]
    image = np.array(pil_image).copy()

    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pil_gray = pil_image.convert("L")

    for contrast_factor in [0.0, 0.5, 1.0]:
        pil_enhanced = F_pil.adjust_contrast(pil_gray, contrast_factor)
        cv_enhanced = F.adjust_contrast(image_gray, contrast_factor)

        pil_array = np.array(pil_enhanced)
        cv_squeezed = cv_enhanced.squeeze()

        if not np.array_equal(pil_array, cv_squeezed):
            diff = np.abs(pil_array.astype(int) - cv_squeezed.astype(int))
            mismatches.append(
                {
                    "image_idx": i,
                    "contrast": contrast_factor,
                    "max_diff": np.max(diff),
                    "num_diff_pixels": np.sum(diff > 0),
                    "pil_unique": np.unique(pil_array)
                    if contrast_factor == 0
                    else None,
                    "cv_unique": np.unique(cv_squeezed)
                    if contrast_factor == 0
                    else None,
                }
            )

print(f"Found {len(mismatches)} mismatches out of 30 tests")
for m in mismatches[:5]:  # Show first 5
    print(f"\nImage {m['image_idx']}, contrast={m['contrast']}:")
    print(f"  Max diff: {m['max_diff']}")
    print(f"  Num diff pixels: {m['num_diff_pixels']}")
    if m["contrast"] == 0:
        print(f"  PIL unique values: {m['pil_unique']}")
        print(f"  CV unique values: {m['cv_unique']}")
