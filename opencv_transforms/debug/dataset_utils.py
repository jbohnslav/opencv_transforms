"""Utilities for testing with real dataset images.

This module provides functions for loading and testing transforms
with images from popular datasets.
"""

from typing import List
from typing import Optional

import numpy as np
from PIL import Image

from opencv_transforms.debug.utils import analyze_pil_precision_issue
from opencv_transforms.debug.utils import create_contrast_test_summary

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def test_with_dataset_image(dataset_name, num_samples, transform_fn):
    """Test transforms with real dataset images.

    Args:
        dataset_name: Name of the HuggingFace dataset to use
        num_samples: Number of samples to test
        transform_fn: Optional transform function to apply

    Returns:
        list: List of test results
    """
    if not HAS_DATASETS:
        raise ImportError(
            "datasets library is required. Install with: pip install datasets"
        )

    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(
        dataset_name, split="train", streaming=True, cache_dir=".cache/"
    )

    results = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        image = sample["image"]
        print(f"\nTesting sample {i + 1}/{num_samples}")

        if transform_fn is not None:
            result = transform_fn(image)
        else:
            # Default: test contrast
            result = create_contrast_test_summary(image)

        results.append(result)

    return results


def load_test_images(
    sources: Optional[List[str]] = None,
) -> List[Image.Image]:
    """Load a variety of test images from different sources.

    Args:
        sources: List of image sources/paths. If None, uses default test images.

    Returns:
        list: List of PIL Images
    """
    images = []

    if sources is None:
        # Create some synthetic test images
        # Gradient image
        gradient = np.zeros((256, 256), dtype=np.uint8)
        for i in range(256):
            gradient[i, :] = i
        images.append(Image.fromarray(gradient))

        # Random noise
        noise = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        images.append(Image.fromarray(noise))

        # Checkerboard
        checker = np.zeros((200, 200), dtype=np.uint8)
        checker[::20, ::20] = 255
        checker[10::20, 10::20] = 255
        images.append(Image.fromarray(checker))
    else:
        for source in sources:
            if isinstance(source, str):
                images.append(Image.open(source))
            elif isinstance(source, Image.Image):
                images.append(source)
            elif isinstance(source, np.ndarray):
                images.append(Image.fromarray(source))

    return images


def test_precision_across_images(images: List[Image.Image]):
    """Test PIL precision issues across multiple images.

    Args:
        images: List of images to test
    """
    print("Testing PIL precision issues across images...")
    print("=" * 60)

    for i, image in enumerate(images):
        print(f"\nImage {i + 1}:")
        print(f"  Size: {image.size}")
        print(f"  Mode: {image.mode}")
        analyze_pil_precision_issue(image)


def run_comprehensive_debug_suite():
    """Run a comprehensive suite of debug tests.

    This function runs various debug utilities on test images to identify
    and document differences between PIL and OpenCV implementations.
    """
    print("Running comprehensive debug suite...")
    print("=" * 60)

    # Load test images
    images = load_test_images()

    # Test precision issues
    test_precision_across_images(images)

    # Test with real dataset
    try:
        dataset_results = test_with_dataset_image("beans", 3, None)
        print("\nDataset test results:", len(dataset_results))
    except Exception as e:
        print(f"\nSkipping dataset tests: {e}")

    print("\nDebug suite complete!")


if __name__ == "__main__":
    run_comprehensive_debug_suite()
