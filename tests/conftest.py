import random
from typing import List
from typing import Tuple

import numpy as np
import pytest
from datasets import load_dataset
from PIL import Image


@pytest.fixture(scope="session")
def imagenet_dataset():
    """Load ImageNet dataset samples for testing."""
    # Use beans dataset as a fallback since ImageNet requires authentication
    try:
        dataset = load_dataset("beans", split="train", streaming=True)
        print("Using beans dataset for testing")
    except Exception:
        # If beans fails, create synthetic data
        print("Creating synthetic test data")
        return _create_synthetic_images()

    samples = []
    for i, sample in enumerate(dataset):
        if i >= 50:  # Limit to 50 samples for tests
            break
        samples.append(sample["image"])

    return samples


@pytest.fixture(scope="session")
def test_images(imagenet_dataset) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """Convert dataset samples to both PIL and OpenCV format."""
    pil_images = imagenet_dataset
    cv_images = [np.array(img) for img in pil_images]
    return pil_images, cv_images


@pytest.fixture
def single_test_image(test_images) -> Tuple[Image.Image, np.ndarray]:
    """Get a single test image in both PIL and OpenCV format."""
    pil_images, cv_images = test_images
    return pil_images[0], cv_images[0]


@pytest.fixture
def random_test_image(test_images) -> Tuple[Image.Image, np.ndarray]:
    """Get a random test image in both PIL and OpenCV format."""
    pil_images, cv_images = test_images
    idx = random.randint(0, len(pil_images) - 1)
    return pil_images[idx], cv_images[idx]


def _create_synthetic_images(num_images: int = 50) -> List[Image.Image]:
    """Create synthetic test images if real dataset is unavailable."""
    images = []
    for _i in range(num_images):
        # Create random RGB image
        np_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_img = Image.fromarray(np_img)
        images.append(pil_img)
    return images


# Test configuration
TOL = 1e-4  # Tolerance for numerical comparisons

# Transform comparison tolerances
TRANSFORM_RTOL = 1e-5  # Relative tolerance for transform comparisons
TRANSFORM_ATOL = 1e-3  # Absolute tolerance (normalized 0-1 range)
PIXEL_ATOL = 2.0  # Absolute tolerance in pixel values (0-255 range)

# Transform-specific tolerances (override defaults)
TRANSFORM_TOLERANCES = {
    "resize": {
        "rtol": 1e-3,
        "atol": 1e-2,
        "pixel_atol": 120.0,
    },  # High tolerance for interpolation
    "rotation": {
        "rtol": 1e-3,
        "atol": 1e-2,
        "pixel_atol": 120.0,
    },  # High tolerance for interpolation
    "affine": {
        "rtol": 1e-3,
        "atol": 1e-2,
        "pixel_atol": 120.0,
    },  # High tolerance for interpolation
    "crop": {"rtol": 1e-7, "atol": 1e-5, "pixel_atol": 0.1},  # Should be nearly exact
    "flip": {"rtol": 1e-7, "atol": 1e-5, "pixel_atol": 0.1},  # Should be nearly exact
    "pad": {"rtol": 1e-7, "atol": 1e-5, "pixel_atol": 0.1},  # Should be nearly exact
}
