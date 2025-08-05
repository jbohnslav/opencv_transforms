import random
import warnings

import numpy as np
import pytest
import torch
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
        # Both torchvision and opencv_transforms now use torch random for compatibility
        torch.manual_seed(42)
        random.seed(42)
        pil_rotated = pil_transforms.RandomRotation(degrees)(pil_image)

        torch.manual_seed(42)
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

        # Resize to ensure crop will work - use PIL resize for both to ensure identical input
        pil_image = pil_transforms.Resize((min_size, min_size))(pil_image)
        cv_image = np.array(
            pil_image
        )  # Convert PIL result to numpy for OpenCV transforms

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

        # Ensure image is large enough - use PIL resize for both to ensure identical input
        min_size = (
            (crop_size + 50) if isinstance(crop_size, int) else (max(crop_size) + 50)
        )
        pil_image = pil_transforms.Resize((min_size, min_size))(pil_image)
        cv_image = np.array(
            pil_image
        )  # Convert PIL result to numpy for OpenCV transforms

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

        # Check output shapes match expected
        if isinstance(crop_size, int):
            expected_shape = (crop_size, crop_size)
        else:
            expected_shape = crop_size

        assert np.array(pil_cropped).shape[:2] == expected_shape
        assert cv_cropped.shape[:2] == expected_shape

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

        # Use same seed for deterministic comparison
        # Both torchvision and opencv_transforms now use torch random for compatibility
        torch.manual_seed(42)
        pil_transformed = pil_transforms.RandomResizedCrop(size, scale=scale)(pil_image)

        torch.manual_seed(42)
        cv_transformed = transforms.RandomResizedCrop(size, scale=scale)(cv_image)

        # Check output shapes match expected
        expected_shape = (size, size) if isinstance(size, int) else size

        assert np.array(pil_transformed).shape[:2] == expected_shape
        assert cv_transformed.shape[:2] == expected_shape

        # Compare the actual transformed outputs
        tolerances = TRANSFORM_TOLERANCES.get(
            "random_resized_crop", {}
        )  # Use specific tolerances for RandomResizedCrop
        assert_transforms_close(pil_transformed, cv_transformed, **tolerances)

    @pytest.mark.parametrize("size", [(224, 224), (256, 256), (128, 128)])
    def test_scale_deprecated(self, single_test_image, size):
        """Test deprecated Scale transform matches Resize behavior."""
        pil_image, cv_image = single_test_image

        # Test that Scale produces the same result as Resize
        cv_resized = transforms.Resize(size)(cv_image)

        # Test Scale (should produce deprecation warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cv_scaled = transforms.Scale(size)(cv_image)
            # Verify deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "deprecated" in str(w[-1].message)

        # Scale should produce identical results to Resize
        assert np.array_equal(cv_scaled, cv_resized)
        assert cv_scaled.shape[:2] == size

    @pytest.mark.parametrize("padding", [10, (5, 10), (5, 10, 15, 20)])
    @pytest.mark.parametrize("fill", [0, 128, (255, 0, 0)])
    @pytest.mark.parametrize(
        "padding_mode", ["constant", "edge", "reflect", "symmetric"]
    )
    def test_pad(self, single_test_image, padding, fill, padding_mode):
        """Test padding functionality matches PIL implementation."""
        pil_image, cv_image = single_test_image

        # Skip color fill for grayscale images or non-constant modes
        if isinstance(fill, tuple) and (
            len(cv_image.shape) != 3 or padding_mode != "constant"
        ):
            pytest.skip(
                "Color fill only supported for RGB images with constant padding"
            )

        pil_padded = pil_transforms.Pad(padding, fill=fill, padding_mode=padding_mode)(
            pil_image
        )
        cv_padded = transforms.Pad(padding, fill=fill, padding_mode=padding_mode)(
            cv_image
        )

        # Use strict tolerances for padding (should be nearly exact)
        tolerances = TRANSFORM_TOLERANCES.get("pad", {})
        assert_transforms_close(pil_padded, cv_padded, **tolerances)

        # Calculate expected output shape
        if isinstance(padding, int):
            expected_h = cv_image.shape[0] + 2 * padding
            expected_w = cv_image.shape[1] + 2 * padding
        elif len(padding) == 2:
            expected_h = cv_image.shape[0] + 2 * padding[1]
            expected_w = cv_image.shape[1] + 2 * padding[0]
        else:  # len(padding) == 4
            expected_h = cv_image.shape[0] + padding[1] + padding[3]
            expected_w = cv_image.shape[1] + padding[0] + padding[2]

        assert cv_padded.shape[0] == expected_h
        assert cv_padded.shape[1] == expected_w

    @pytest.mark.parametrize("crop_size", [100, (100, 150)])
    @pytest.mark.parametrize("vertical_flip", [False, True])
    def test_ten_crop(self, single_test_image, crop_size, vertical_flip):
        """Test ten crop functionality."""
        pil_image, cv_image = single_test_image

        # Ensure image is large enough for cropping
        min_size = crop_size if isinstance(crop_size, int) else max(crop_size)
        if min(cv_image.shape[:2]) < min_size + 50:
            # Resize image to be large enough
            resize_size = min_size + 100
            pil_image = pil_transforms.Resize((resize_size, resize_size))(pil_image)
            cv_image = transforms.Resize((resize_size, resize_size))(cv_image)

        pil_crops = pil_transforms.TenCrop(crop_size, vertical_flip=vertical_flip)(
            pil_image
        )
        cv_crops = transforms.TenCrop(crop_size, vertical_flip=vertical_flip)(cv_image)

        # Should return exactly 10 crops
        assert len(pil_crops) == 10
        assert len(cv_crops) == 10

        # Each crop should have the expected size
        expected_shape = (
            (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        )

        for _i, (pil_crop, cv_crop) in enumerate(zip(pil_crops, cv_crops)):
            assert np.array(pil_crop).shape[:2] == expected_shape
            assert cv_crop.shape[:2] == expected_shape

            # Compare each crop with appropriate tolerances
            tolerances = TRANSFORM_TOLERANCES.get("crop", {})
            assert_transforms_close(pil_crop, cv_crop, **tolerances)
