#!/usr/bin/env python3
"""Quick test script to validate multi-size data loading functions."""

import sys

sys.path.append(".")

from benchmark import load_cityscapes_dataset
from benchmark import load_imagenet_validation
from benchmark import prepare_multi_size_images


def test_data_loading():
    """Test the new data loading functions."""
    print("=" * 60)
    print("TESTING MULTI-SIZE DATA LOADING")
    print("=" * 60)

    # Test with small sample sizes for speed
    print("\n1. Testing Cityscapes loading...")
    cityscapes_samples = load_cityscapes_dataset(num_samples=5)
    print(f"   Loaded {len(cityscapes_samples)} Cityscapes samples")
    if cityscapes_samples:
        img = cityscapes_samples[0]
        print(f"   First image size: {img.size} (W x H)")

    print("\n2. Testing ImageNet loading...")
    imagenet_samples = load_imagenet_validation(num_samples=5)
    print(f"   Loaded {len(imagenet_samples)} ImageNet samples")
    if imagenet_samples:
        img = imagenet_samples[0]
        print(f"   First image size: {img.size} (W x H)")

    print("\n3. Testing multi-size image preparation...")
    size_dict = prepare_multi_size_images(cityscapes_samples, imagenet_samples)

    print(f"   Prepared {len(size_dict)} different sizes:")
    for (w, h), (pil_imgs, cv_imgs) in size_dict.items():
        print(f"   - {w}x{h}: {len(pil_imgs)} PIL images, {len(cv_imgs)} CV arrays")
        if pil_imgs:
            actual_size = pil_imgs[0].size
            print(f"     Verified size: {actual_size}")

    print("\n✅ Data loading test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_loading()
