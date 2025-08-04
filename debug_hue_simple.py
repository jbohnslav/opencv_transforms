import numpy as np
from PIL import Image
from torchvision.transforms import functional as F_pil

# Create a simple test image
pil_image = Image.new("RGB", (10, 10), (100, 150, 200))
print("Original PIL image array:")
print(np.array(pil_image)[0, 0])

# Apply hue adjustment with factor 0.0
pil_adjusted = F_pil.adjust_hue(pil_image, 0.0)
print("\nPIL adjusted with hue_factor=0.0:")
print(np.array(pil_adjusted)[0, 0])

print("\nAre they equal?", np.array_equal(np.array(pil_image), np.array(pil_adjusted)))
