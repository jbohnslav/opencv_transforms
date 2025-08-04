import random

import numpy as np
import pytest
from conftest import TRANSFORM_TOLERANCES
from torchvision import transforms as pil_transforms
from utils import assert_transforms_close

from opencv_transforms import transforms


class TestSpatialTransforms:
    """Test spatial transformations (resize, rotation, cropping, etc.)"""

    @pytest.mark.parametrize("size", [(224, 224), (256, 256), (128, 128)])
    def test_resize(self, single_test_image, size):
        """Test image resizing matches PIL implementation."""
        pil_image, cv_image = single_test_image

        pil_resized = pil_transforms.Resize(size)(pil_image)
        cv_resized = transforms.Resize(size)(cv_image)

        # Use new assertion with appropriate tolerances for resize
        tolerances = TRANSFORM_TOLERANCES.get("resize", {})
        assert_transforms_close(pil_resized, cv_resized, **tolerances)

        # Check output shapes
        assert np.array(pil_resized).shape[:2] == size
        assert cv_resized.shape[:2] == size

    @pytest.mark.parametrize("degrees", [10, 30, 45])
    def test_rotation(self, single_test_image, degrees):
        """Test image rotation."""
        pil_image, cv_image = single_test_image

        # Use fixed seed for deterministic results
        random.seed(42)
        pil_rotated = pil_transforms.RandomRotation(degrees)(pil_image)

        random.seed(42)
        cv_rotated = transforms.RandomRotation(degrees)(cv_image)

        # Use new assertion with appropriate tolerances for rotation
        tolerances = TRANSFORM_TOLERANCES.get("rotation", {})
        assert_transforms_close(pil_rotated, cv_rotated, **tolerances)

    @pytest.mark.parametrize("crop_size", [224, (224, 224), (200, 300)])
    def test_five_crop(self, single_test_image, crop_size):
        """Test five crop transformation."""
        pil_image, cv_image = single_test_image

        # Ensure image is large enough for cropping
        min_size = crop_size + 50 if isinstance(crop_size, int) else max(crop_size) + 50

        # Resize to ensure crop will work
        pil_image = pil_transforms.Resize((min_size, min_size))(pil_image)
        cv_image = transforms.Resize((min_size, min_size))(cv_image)

        pil_crops = pil_transforms.FiveCrop(crop_size)(pil_image)
        cv_crops = transforms.FiveCrop(crop_size)(cv_image)

        # Check we get 5 crops
        assert len(pil_crops) == 5
        assert len(cv_crops) == 5

        # Compare each crop
        for pil_crop, cv_crop in zip(pil_crops, cv_crops):
            tolerances = TRANSFORM_TOLERANCES.get("crop", {})
            assert_transforms_close(pil_crop, cv_crop, **tolerances)

    @pytest.mark.parametrize("crop_size", [224, (224, 224)])
    def test_center_crop(self, single_test_image, crop_size):
        """Test center crop transformation."""
        pil_image, cv_image = single_test_image

        # Ensure image is large enough
        min_size = (
            (crop_size + 50) if isinstance(crop_size, int) else (max(crop_size) + 50)
        )
        pil_image = pil_transforms.Resize((min_size, min_size))(pil_image)
        cv_image = transforms.Resize((min_size, min_size))(cv_image)

        pil_cropped = pil_transforms.CenterCrop(crop_size)(pil_image)
        cv_cropped = transforms.CenterCrop(crop_size)(cv_image)

        tolerances = TRANSFORM_TOLERANCES.get("crop", {})
        assert_transforms_close(pil_cropped, cv_cropped, **tolerances)

    @pytest.mark.parametrize("crop_size", [224, (224, 224)])
    def test_random_crop(self, single_test_image, crop_size):
        """Test random crop transformation."""
        pil_image, cv_image = single_test_image

        # Ensure image is large enough
        min_size = (
            (crop_size + 50) if isinstance(crop_size, int) else (max(crop_size) + 50)
        )
        pil_image = pil_transforms.Resize((min_size, min_size))(pil_image)
        cv_image = transforms.Resize((min_size, min_size))(cv_image)

        # Use same seed for deterministic comparison
        random.seed(42)
        pil_cropped = pil_transforms.RandomCrop(crop_size)(pil_image)

        random.seed(42)
        cv_cropped = transforms.RandomCrop(crop_size)(cv_image)

        # Check output dimensions
        expected_size = (
            (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        )
        assert np.array(pil_cropped).shape[:2] == expected_size
        assert cv_cropped.shape[:2] == expected_size

    def test_horizontal_flip(self, single_test_image):
        """Test horizontal flip transformation."""
        pil_image, cv_image = single_test_image

        pil_flipped = pil_transforms.RandomHorizontalFlip(p=1.0)(pil_image)
        cv_flipped = transforms.RandomHorizontalFlip(p=1.0)(cv_image)

        tolerances = TRANSFORM_TOLERANCES.get("flip", {})
        assert_transforms_close(pil_flipped, cv_flipped, **tolerances)

    def test_vertical_flip(self, single_test_image):
        """Test vertical flip transformation."""
        pil_image, cv_image = single_test_image

        pil_flipped = pil_transforms.RandomVerticalFlip(p=1.0)(pil_image)
        cv_flipped = transforms.RandomVerticalFlip(p=1.0)(cv_image)

        tolerances = TRANSFORM_TOLERANCES.get("flip", {})
        assert_transforms_close(pil_flipped, cv_flipped, **tolerances)

    @pytest.mark.parametrize("scale", [(0.5, 1.0), (0.8, 1.2)])
    @pytest.mark.parametrize("size", [224, (224, 224)])
    def test_random_resized_crop(self, single_test_image, scale, size):
        """Test random resized crop transformation."""
        pil_image, cv_image = single_test_image

        # Use same seed for comparison
        random.seed(42)
        pil_transformed = pil_transforms.RandomResizedCrop(size, scale=scale)(pil_image)

        random.seed(42)
        cv_transformed = transforms.RandomResizedCrop(size, scale=scale)(cv_image)

        # Check output dimensions
        expected_size = (size, size) if isinstance(size, int) else size
        assert np.array(pil_transformed).shape[:2] == expected_size
        assert cv_transformed.shape[:2] == expected_size
