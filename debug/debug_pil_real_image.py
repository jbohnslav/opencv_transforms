import numpy as np
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import functional as F_pil

# Load the test image
dataset = load_dataset("beans", split="test", cache_dir="tests/.cache/")
pil_image = dataset[0]["image"]
pil_gray = pil_image.convert("L")

# Get the numpy array
gray_array = np.array(pil_gray)

# Apply contrast=1.0
pil_enhanced = F_pil.adjust_contrast(pil_gray, 1.0)
enhanced_array = np.array(pil_enhanced)

# Check specific problematic pixels
problematic = [(135, 427), (278, 2), (307, 420), (309, 24), (358, 42), (464, 170)]

print("Checking problematic pixels:")
for r, c in problematic:
    orig = gray_array[r, c]
    enh = enhanced_array[r, c]
    if orig != enh:
        print(f"  ({r},{c}): {orig} -> {enh} (changed by {enh - orig})")

# Check if PIL is doing some color space conversion
print("\nChecking PIL image properties:")
print(f"Original mode: {pil_gray.mode}")
print(f"Enhanced mode: {pil_enhanced.mode}")

# Save both images and reload to check for any conversion artifacts
import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    orig_path = os.path.join(tmpdir, "orig.png")
    enh_path = os.path.join(tmpdir, "enh.png")

    pil_gray.save(orig_path)
    pil_enhanced.save(enh_path)

    # Reload and compare
    reloaded_orig = np.array(Image.open(orig_path))
    reloaded_enh = np.array(Image.open(enh_path))

    print(
        f"\nAfter save/reload, are they equal? {np.array_equal(reloaded_orig, reloaded_enh)}"
    )

# Direct PIL ImageEnhance test
from PIL import ImageEnhance

enhancer = ImageEnhance.Contrast(pil_gray)
direct_enhanced = enhancer.enhance(1.0)
direct_array = np.array(direct_enhanced)

print(
    f"\nDirect ImageEnhance, are they equal? {np.array_equal(gray_array, direct_array)}"
)
diff_count = np.sum(gray_array != direct_array)
print(f"Number of different pixels: {diff_count}")
