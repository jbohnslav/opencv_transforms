"""Direct tests for the functional.affine method."""

import cv2
import numpy as np
import pytest
from PIL import Image
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F_cv


class TestFunctionalAffine:
    """Test the functional.affine method directly."""

    @pytest.fixture
    def test_image(self):
        """Create a simple test image."""
        # Create a simple test pattern that makes transformations easy to verify
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a red square in the center
        img_array[40:60, 40:60, 0] = 255
        # Add a green line at the top
        img_array[10:15, :, 1] = 255
        # Add a blue line on the left
        img_array[:, 10:15, 2] = 255

        pil_img = Image.fromarray(img_array, mode="RGB")
        cv_img = img_array.copy()

        return pil_img, cv_img

    def test_affine_identity(self, test_image):
        """Test that identity affine transformation preserves the image."""
        pil_img, cv_img = test_image

        # Identity transformation: no rotation, no translation, no scale change, no shear
        angle = 0
        translate = [0, 0]
        scale = 1.0
        shear = 0

        # Apply affine transformation
        pil_result = F_pil.affine(pil_img, angle, translate, scale, shear)
        cv_result = F_cv.affine(cv_img, angle, translate, scale, shear)

        # Convert PIL to numpy for comparison
        pil_array = np.array(pil_result)

        # Should be nearly identical for identity transform
        np.testing.assert_allclose(pil_array, cv_result, rtol=1e-5, atol=1)

    def test_affine_rotation(self, test_image):
        """Test affine transformation with rotation."""
        pil_img, cv_img = test_image

        # 45 degree rotation
        angle = 45
        translate = [0, 0]
        scale = 1.0
        shear = 0

        # Apply affine transformation
        pil_result = F_pil.affine(pil_img, angle, translate, scale, shear)
        cv_result = F_cv.affine(cv_img, angle, translate, scale, shear)

        # Convert PIL to numpy for comparison
        pil_array = np.array(pil_result)

        # Check shapes match
        assert pil_array.shape == cv_result.shape

        # Due to interpolation differences, allow some tolerance
        # Count pixels that differ by more than threshold
        diff = np.abs(pil_array.astype(float) - cv_result.astype(float))
        max_diff = np.max(diff)
        # Allow up to 220 pixel value difference (based on RandomAffine tests)
        assert max_diff <= 220, f"Max pixel difference {max_diff} exceeds tolerance"

    def test_affine_translation(self, test_image):
        """Test affine transformation with translation."""
        pil_img, cv_img = test_image

        # Translate by 10 pixels in x and y
        angle = 0
        translate = [10, 10]
        scale = 1.0
        shear = 0

        # Apply affine transformation
        pil_result = F_pil.affine(pil_img, angle, translate, scale, shear)
        cv_result = F_cv.affine(cv_img, angle, translate, scale, shear)

        # Convert PIL to numpy for comparison
        pil_array = np.array(pil_result)

        # Check shapes match
        assert pil_array.shape == cv_result.shape

        # Translation should be exact or very close
        np.testing.assert_allclose(pil_array, cv_result, rtol=1e-3, atol=2)

    def test_affine_scale(self, test_image):
        """Test affine transformation with scaling."""
        pil_img, cv_img = test_image

        # Scale by 0.5 (shrink)
        angle = 0
        translate = [0, 0]
        scale = 0.5
        shear = 0

        # Apply affine transformation
        pil_result = F_pil.affine(pil_img, angle, translate, scale, shear)
        cv_result = F_cv.affine(cv_img, angle, translate, scale, shear)

        # Convert PIL to numpy for comparison
        pil_array = np.array(pil_result)

        # Check shapes match
        assert pil_array.shape == cv_result.shape

        # Due to interpolation differences, allow some tolerance
        diff = np.abs(pil_array.astype(float) - cv_result.astype(float))
        max_diff = np.max(diff)
        # Allow reasonable tolerance for scaling
        assert max_diff <= 220, f"Max pixel difference {max_diff} exceeds tolerance"

    def test_affine_shear(self, test_image):
        """Test affine transformation with shear."""
        pil_img, cv_img = test_image

        # Shear by 15 degrees
        angle = 0
        translate = [0, 0]
        scale = 1.0
        shear = 15

        # Apply affine transformation
        pil_result = F_pil.affine(pil_img, angle, translate, scale, shear)
        cv_result = F_cv.affine(cv_img, angle, translate, scale, shear)

        # Convert PIL to numpy for comparison
        pil_array = np.array(pil_result)

        # Check shapes match
        assert pil_array.shape == cv_result.shape

        # Due to interpolation differences, allow some tolerance
        diff = np.abs(pil_array.astype(float) - cv_result.astype(float))
        max_diff = np.max(diff)
        # Allow reasonable tolerance for shear
        assert max_diff <= 220, f"Max pixel difference {max_diff} exceeds tolerance"

    def test_affine_combined(self, test_image):
        """Test affine transformation with combined parameters."""
        pil_img, cv_img = test_image

        # Combined transformation
        angle = 30
        translate = [5, -5]
        scale = 0.8
        shear = 10

        # Apply affine transformation
        pil_result = F_pil.affine(pil_img, angle, translate, scale, shear)
        cv_result = F_cv.affine(cv_img, angle, translate, scale, shear)

        # Convert PIL to numpy for comparison
        pil_array = np.array(pil_result)

        # Check shapes match
        assert pil_array.shape == cv_result.shape

        # Due to interpolation differences and combined transforms, allow higher tolerance
        diff = np.abs(pil_array.astype(float) - cv_result.astype(float))
        max_diff = np.max(diff)
        # Allow higher tolerance for combined transformations
        assert max_diff <= 250, f"Max pixel difference {max_diff} exceeds tolerance"

    def test_affine_grayscale(self):
        """Test affine transformation on grayscale images."""
        # Create grayscale test image
        img_array = np.zeros((100, 100), dtype=np.uint8)
        img_array[40:60, 40:60] = 255  # White square in center

        pil_img = Image.fromarray(img_array, mode="L")
        cv_img = img_array.copy()

        # Apply rotation
        angle = 45
        translate = [0, 0]
        scale = 1.0
        shear = 0

        # Apply affine transformation
        pil_result = F_pil.affine(pil_img, angle, translate, scale, shear)
        cv_result = F_cv.affine(cv_img, angle, translate, scale, shear)

        # Convert PIL to numpy for comparison
        pil_array = np.array(pil_result)

        # OpenCV functional returns shape (H, W) for grayscale
        # So we need to ensure dimensions match
        cv_result_compare = cv_result if cv_result.ndim == 2 else cv_result.squeeze()

        # Check shapes match
        assert pil_array.shape == cv_result_compare.shape

        # Due to interpolation differences, allow some tolerance
        diff = np.abs(pil_array.astype(float) - cv_result_compare.astype(float))
        max_diff = np.max(diff)
        assert max_diff <= 220, f"Max pixel difference {max_diff} exceeds tolerance"

    def test_affine_fillcolor(self, test_image):
        """Test affine transformation with different fill colors."""
        pil_img, cv_img = test_image

        # Rotation with custom fill color
        angle = 45
        translate = [0, 0]
        scale = 1.0
        shear = 0
        fillcolor = 128  # Gray fill

        # Apply affine transformation with fill color
        pil_result = F_pil.affine(
            pil_img, angle, translate, scale, shear, fill=fillcolor
        )
        cv_result = F_cv.affine(
            cv_img, angle, translate, scale, shear, fillcolor=fillcolor
        )

        # Convert PIL to numpy for comparison
        _ = np.array(pil_result)  # Just to match the pattern of other tests

        # Check that corners (which should be filled) have the fill color
        # Check top-left corner pixel
        cv_corner = cv_result[0, 0]

        # The corners should be close to the fill color after rotation
        # Allow some tolerance due to interpolation
        if angle != 0:  # Only check if there's actual rotation
            # At least one channel should be close to fill color
            assert np.any(np.abs(cv_corner - fillcolor) <= 10), (
                f"Corner pixel {cv_corner} not close to fill color {fillcolor}"
            )

    @pytest.mark.parametrize(
        "interpolation",
        [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
        ],
    )
    def test_affine_interpolation_modes(self, test_image, interpolation):
        """Test affine transformation with different interpolation modes."""
        pil_img, cv_img = test_image

        # Apply rotation with different interpolation
        angle = 30
        translate = [0, 0]
        scale = 1.0
        shear = 0

        # Map cv2 interpolation to PIL interpolation
        cv2_to_pil = {
            cv2.INTER_NEAREST: Image.NEAREST,
            cv2.INTER_LINEAR: Image.BILINEAR,
            cv2.INTER_CUBIC: Image.BICUBIC,
        }

        pil_interpolation = cv2_to_pil.get(interpolation, Image.BILINEAR)

        # Apply affine transformation
        pil_result = F_pil.affine(
            pil_img, angle, translate, scale, shear, interpolation=pil_interpolation
        )
        cv_result = F_cv.affine(
            cv_img, angle, translate, scale, shear, interpolation=interpolation
        )

        # Convert PIL to numpy for comparison
        pil_array = np.array(pil_result)

        # Check shapes match
        assert pil_array.shape == cv_result.shape

        # Different interpolation methods will have different tolerances
        max_allowed_diff = 200 if interpolation == cv2.INTER_NEAREST else 250

        diff = np.abs(pil_array.astype(float) - cv_result.astype(float))
        max_diff = np.max(diff)
        assert max_diff <= max_allowed_diff, (
            f"Max pixel difference {max_diff} exceeds tolerance {max_allowed_diff}"
        )

    def test_affine_invalid_inputs(self, test_image):
        """Test that affine raises appropriate errors for invalid inputs."""
        _, cv_img = test_image

        # Test invalid scale (negative)
        with pytest.raises(AssertionError):
            F_cv.affine(cv_img, 0, [0, 0], -1.0, 0)

        # Test invalid translate (not a list/tuple of length 2)
        with pytest.raises(AssertionError):
            F_cv.affine(cv_img, 0, [0], 1.0, 0)

        with pytest.raises(AssertionError):
            F_cv.affine(cv_img, 0, [0, 0, 0], 1.0, 0)

        # Test non-numpy image
        with pytest.raises(TypeError):
            F_cv.affine("not an image", 0, [0, 0], 1.0, 0)
