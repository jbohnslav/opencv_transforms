"""Simplified benchmark script without dataset loading."""

import random
import time
from typing import Callable
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as pil_transforms

from opencv_transforms import transforms as cv_transforms

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("Creating synthetic test images...")
# Create synthetic images for testing
pil_images = [
    Image.fromarray(np.random.randint(0, 255, (500, 375, 3), dtype=np.uint8))
    for _ in range(20)
]
cv_images = [np.array(img) for img in pil_images]

print(f"Created {len(pil_images)} test images")


def benchmark_transform(transform_func: Callable, images: List):
    """Time transform on images."""
    times = []
    for img in images:
        start = time.time()
        try:
            result = transform_func(img)
            if isinstance(result, tuple):
                pass  # FiveCrop, TenCrop return tuples
        except Exception as e:
            print(f"  Warning: {e}")
            continue
        times.append(time.time() - start)

    if not times:
        return 0.0
    return np.mean(times)


# Test class-based transforms
print("\n" + "=" * 60)
print("BENCHMARKING CLASS-BASED TRANSFORMS")
print("=" * 60)

transforms_to_test = [
    # Basic transforms
    ("ToTensor", pil_transforms.ToTensor(), cv_transforms.ToTensor()),
    (
        "Resize (256x256)",
        pil_transforms.Resize((256, 256)),
        cv_transforms.Resize((256, 256)),
    ),
    ("CenterCrop (224)", pil_transforms.CenterCrop(224), cv_transforms.CenterCrop(224)),
    ("RandomCrop (224)", pil_transforms.RandomCrop(224), cv_transforms.RandomCrop(224)),
    (
        "RandomHorizontalFlip",
        pil_transforms.RandomHorizontalFlip(p=1.0),
        cv_transforms.RandomHorizontalFlip(p=1.0),
    ),
    (
        "RandomVerticalFlip",
        pil_transforms.RandomVerticalFlip(p=1.0),
        cv_transforms.RandomVerticalFlip(p=1.0),
    ),
    ("Pad (10)", pil_transforms.Pad(10), cv_transforms.Pad(10)),
    ("Grayscale (3ch)", pil_transforms.Grayscale(3), cv_transforms.Grayscale(3)),
    (
        "RandomGrayscale",
        pil_transforms.RandomGrayscale(p=1.0),
        cv_transforms.RandomGrayscale(p=1.0),
    ),
    (
        "RandomRotation (10)",
        pil_transforms.RandomRotation(10),
        cv_transforms.RandomRotation(10),
    ),
    (
        "RandomAffine (10)",
        pil_transforms.RandomAffine(10),
        cv_transforms.RandomAffine(10),
    ),
    (
        "RandomResizedCrop",
        pil_transforms.RandomResizedCrop(224),
        cv_transforms.RandomResizedCrop(224),
    ),
    (
        "ColorJitter (br=0.2)",
        pil_transforms.ColorJitter(brightness=0.2),
        cv_transforms.ColorJitter(brightness=0.2),
    ),
    ("FiveCrop (224)", pil_transforms.FiveCrop(224), cv_transforms.FiveCrop(224)),
    ("TenCrop (224)", pil_transforms.TenCrop(224), cv_transforms.TenCrop(224)),
]

results = []
for name, pil_transform, cv_transform in transforms_to_test:
    print(f"\n{name}:")

    pil_time = benchmark_transform(pil_transform, pil_images)
    cv_time = benchmark_transform(cv_transform, cv_images)

    speedup = pil_time / cv_time if cv_time > 0 else 0

    print(f"  PIL: {pil_time * 1000:.3f} ms")
    print(f"  OpenCV: {cv_time * 1000:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    results.append(
        {
            "transform": name,
            "pil_time": pil_time * 1000,
            "cv_time": cv_time * 1000,
            "speedup": speedup,
        }
    )

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Transform':<25} {'PIL (ms)':<12} {'CV (ms)':<12} {'Speedup':<10}")
print("-" * 60)

for r in results:
    print(
        f"{r['transform']:<25} {r['pil_time']:<12.3f} {r['cv_time']:<12.3f} {r['speedup']:<10.2f}x"
    )

avg_speedup = np.mean([r["speedup"] for r in results])
print("-" * 60)
print(f"{'Average Speedup:':<25} {'':<12} {'':<12} {avg_speedup:<10.2f}x")

print("\nBenchmark completed successfully!")
