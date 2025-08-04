import numpy as np
from datasets import load_dataset
from PIL import ImageEnhance

# Load first image from dataset
dataset = load_dataset("beans", split="train", streaming=True)
sample = next(iter(dataset))
pil_image = sample["image"]

# Convert to grayscale
pil_gray = pil_image.convert("L")

# Apply contrast=0 using PIL
enhancer = ImageEnhance.Contrast(pil_gray)
pil_result = enhancer.enhance(0.0)

# Check the result
pil_array = np.array(pil_result)
print("PIL result unique values:", np.unique(pil_array))
print("First value:", pil_array[0, 0])

# Calculate mean ourselves
gray_array = np.array(pil_gray)
mean_val = np.mean(gray_array)
print(f"\nMean of grayscale image: {mean_val}")
print(f"Rounded mean: {round(mean_val)}")
print(f"Int mean: {int(mean_val)}")

# PIL seems to use a degenerate image - let's check
# PIL's ImageEnhance.Contrast creates a degenerate image by converting to L mode
# and filling with the mean
print("\nChecking PIL's degenerate image calculation...")
# PIL calculates mean differently - it uses the histogram
hist = pil_gray.histogram()
n = sum(hist)
mean_pil = sum(i * w for i, w in enumerate(hist)) / n
print(f"PIL histogram-based mean: {mean_pil}")
print(
    f"PIL histogram-based mean (int): {int(mean_pil + 0.5)}"
)  # PIL rounds by adding 0.5
