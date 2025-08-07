"""Quick test of benchmark functionality with minimal transforms."""

import time

import numpy as np
from PIL import Image
from torchvision import transforms as pil_transforms

from opencv_transforms import transforms as cv_transforms

# Create test images
print("Creating test images...")
pil_images = [
    Image.fromarray(np.random.randint(0, 255, (500, 375, 3), dtype=np.uint8))
    for _ in range(5)
]
cv_images = [np.array(img) for img in pil_images]

# Test basic transforms
transforms_to_test = [
    {
        "name": "Resize",
        "pil": pil_transforms.Resize((256, 256)),
        "cv": cv_transforms.Resize((256, 256)),
    },
    {
        "name": "CenterCrop",
        "pil": pil_transforms.CenterCrop(224),
        "cv": cv_transforms.CenterCrop(224),
    },
    {
        "name": "ToTensor",
        "pil": pil_transforms.ToTensor(),
        "cv": cv_transforms.ToTensor(),
    },
]

for config in transforms_to_test:
    print(f"\n{config['name']}:")

    # Test PIL
    start = time.time()
    for img in pil_images:
        _ = config["pil"](img)
    pil_time = time.time() - start

    # Test OpenCV
    start = time.time()
    for img in cv_images:
        _ = config["cv"](img)
    cv_time = time.time() - start

    print(f"  PIL: {pil_time * 1000:.2f}ms")
    print(f"  OpenCV: {cv_time * 1000:.2f}ms")
    print(f"  Speedup: {pil_time / cv_time:.2f}x")

print("\nAll tests completed successfully!")
