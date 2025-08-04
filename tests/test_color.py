import random

import cv2
import numpy as np
import pytest
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F


class TestContrast:
    @pytest.mark.parametrize("random_seed", [1, 2, 3, 4])
    @pytest.mark.parametrize("contrast_factor", [0.0, 0.5, 1.0, 2.0])
    def test_contrast(self, test_images, contrast_factor, random_seed):
        """Test contrast adjustment matches PIL implementation."""
        random.seed(random_seed)
        pil_images, cv_images = test_images

        # Select random image
        idx = random.randint(0, len(pil_images) - 1)
        pil_image = pil_images[idx]
        image = np.array(pil_image).copy()

        pil_enhanced = F_pil.adjust_contrast(pil_image, contrast_factor)
        np_enhanced = F.adjust_contrast(image, contrast_factor)

        # Allow small differences (±1 pixel value) due to precision differences in
        # RGB-to-grayscale conversion. Originally failed with exact equality due to:
        # - PIL uses pure floating-point: (299*R + 587*G + 114*B) / 1000
        # - OpenCV uses optimized integer arithmetic with different rounding
        # - Small mean differences (e.g., 134.428 vs 134.432) cause systematic ±1 pixel diffs
        pil_array = np.array(pil_enhanced)
        cv_array = np_enhanced.squeeze()
        diff = np.abs(pil_array.astype(float) - cv_array.astype(float))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Tolerance based on empirical analysis: max ±1 pixel, affects ~50% of pixels
        assert max_diff <= 1.0, (
            f"Max difference {max_diff} exceeds tolerance of 1 pixel"
        )
        assert mean_diff < 1.0, f"Mean difference {mean_diff} exceeds tolerance of 1.0"

    @pytest.mark.parametrize("n_images", [1, 11])
    def test_multichannel_contrast(self, single_test_image, n_images):
        """Test contrast adjustment works on multichannel images."""
        pil_image, _ = single_test_image
        image = np.array(pil_image).copy()
        contrast_factor = 0.1

        multichannel_image = np.concatenate([image for _ in range(n_images)], axis=-1)
        # This should not raise an exception (was fixed in recent versions)
        np_enhanced = F.adjust_contrast(multichannel_image, contrast_factor)
        assert np_enhanced is not None

    @pytest.mark.parametrize("contrast_factor", [0, 0.5, 1.0])
    def test_grayscale_contrast(self, single_test_image, contrast_factor):
        """Test contrast adjustment on grayscale images."""
        pil_image, _ = single_test_image
        image = np.array(pil_image).copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Convert PIL image to grayscale
        pil_image = pil_image.convert("L")

        pil_enhanced = F_pil.adjust_contrast(pil_image, contrast_factor)
        np_enhanced = F.adjust_contrast(image, contrast_factor)

        # Allow differences due to comparing PIL vs OpenCV grayscale conversion methods:
        # - PIL convert("L"): Uses (299*R + 587*G + 114*B) / 1000 with pixel rounding
        # - OpenCV cv2.cvtColor(COLOR_RGB2GRAY): Same weights, different implementation
        pil_array = np.array(pil_enhanced)
        cv_array = np_enhanced.squeeze()
        diff = np.abs(pil_array.astype(float) - cv_array.astype(float))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Tolerance reflects that we're comparing different grayscale conversions + contrast
        assert max_diff <= 1.0, (
            f"Max difference {max_diff} exceeds tolerance of 1 pixel"
        )
        assert mean_diff < 1.0, f"Mean difference {mean_diff} exceeds tolerance of 1.0"


class TestBrightness:
    @pytest.mark.parametrize("brightness_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_brightness_adjustment(self, single_test_image, brightness_factor):
        """Test brightness adjustment matches PIL implementation."""
        pil_image, cv_image = single_test_image

        pil_enhanced = F_pil.adjust_brightness(pil_image, brightness_factor)
        cv_enhanced = F.adjust_brightness(cv_image, brightness_factor)

        # Allow small differences due to different implementations
        diff = np.abs(np.array(pil_enhanced).astype(float) - cv_enhanced.astype(float))
        assert np.mean(diff) < 1.0  # Mean difference less than 1 pixel value


class TestColorTransforms:
    def test_grayscale_conversion(self, single_test_image):
        """Test grayscale conversion."""
        pil_image, cv_image = single_test_image

        pil_gray = F_pil.to_grayscale(pil_image, num_output_channels=3)
        cv_gray = F.to_grayscale(cv_image, num_output_channels=3)

        # Check shapes match
        assert np.array(pil_gray).shape == cv_gray.shape

        # Check conversion produces similar results
        diff = np.abs(np.array(pil_gray).astype(float) - cv_gray.astype(float))
        assert np.mean(diff) < 5.0  # Allow some difference in grayscale algorithms

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5, 2.0])
    def test_gamma_adjustment(self, single_test_image, gamma):
        """Test gamma adjustment."""
        _, cv_image = single_test_image

        # Test that gamma adjustment doesn't crash and produces reasonable output
        gamma_adjusted = F.adjust_gamma(cv_image, gamma)

        assert gamma_adjusted.shape == cv_image.shape
        assert gamma_adjusted.dtype == cv_image.dtype
        assert 0 <= gamma_adjusted.min() <= gamma_adjusted.max() <= 255
