import random

import numpy as np
import pytest
import torch
from torchvision import transforms as T

import opencv_transforms.transforms as cv_transforms


class TestRandomApply:
    @pytest.mark.parametrize("p", [0.0, 0.3, 0.5, 0.7, 1.0])
    @pytest.mark.parametrize("random_seed", [1, 2, 3, 42, 123])
    def test_random_apply_single_transform(self, test_images, p, random_seed):
        """Test RandomApply with single transform matches PyTorch behavior."""
        pil_images, cv_images = test_images

        # Select random image
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        idx = random.randint(0, len(pil_images) - 1)
        pil_image = pil_images[idx]
        cv_image = cv_images[idx].copy()

        # Create transforms with deterministic parameters
        pil_transform = T.RandomApply([T.ColorJitter(brightness=0.5)], p=p)
        cv_transform = cv_transforms.RandomApply(
            [cv_transforms.ColorJitter(brightness=0.5)], p=p
        )

        # Apply with same random seed
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        pil_result = pil_transform(pil_image)

        torch.manual_seed(random_seed)
        random.seed(random_seed)
        cv_result = cv_transform(cv_image)

        # Check if both applied or both didn't apply
        pil_applied = not np.array_equal(np.array(pil_image), np.array(pil_result))
        cv_applied = not np.array_equal(cv_image, cv_result)

        assert pil_applied == cv_applied, (
            f"Application mismatch at p={p}, seed={random_seed}: PyTorch={pil_applied}, OpenCV={cv_applied}"
        )

        # If both applied, the results should be similar (allowing for precision differences)
        if pil_applied and cv_applied:
            pil_array = np.array(pil_result)
            # Allow generous tolerance for color jitter differences between PIL and OpenCV
            # ColorJitter can produce significantly different results due to different algorithms
            assert np.allclose(pil_array, cv_result, rtol=0.3, atol=50), (
                "Applied transforms should produce reasonably similar results"
            )

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_random_apply_multiple_transforms(self, test_images, p, random_seed):
        """Test RandomApply with multiple transforms matches PyTorch behavior."""
        pil_images, cv_images = test_images

        # Select random image
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        idx = random.randint(0, len(pil_images) - 1)
        pil_image = pil_images[idx]
        cv_image = cv_images[idx].copy()

        # Create multiple transform pipeline
        transforms_list = [
            T.ColorJitter(brightness=0.3),
            T.RandomHorizontalFlip(p=1.0),  # Always flip for deterministic behavior
        ]
        cv_transforms_list = [
            cv_transforms.ColorJitter(brightness=0.3),
            cv_transforms.RandomHorizontalFlip(p=1.0),
        ]

        pil_transform = T.RandomApply(transforms_list, p=p)
        cv_transform = cv_transforms.RandomApply(cv_transforms_list, p=p)

        # Apply with same random seed
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        pil_result = pil_transform(pil_image)

        torch.manual_seed(random_seed)
        random.seed(random_seed)
        cv_result = cv_transform(cv_image)

        # Check if both applied or both didn't apply
        pil_applied = not np.array_equal(np.array(pil_image), np.array(pil_result))
        cv_applied = not np.array_equal(cv_image, cv_result)

        assert pil_applied == cv_applied, (
            f"Application mismatch at p={p}, seed={random_seed}: PyTorch={pil_applied}, OpenCV={cv_applied}"
        )

    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_random_apply_probability_zero(self, test_images, random_seed):
        """Test RandomApply with p=0.0 never applies transforms."""
        pil_images, cv_images = test_images

        # Select random image
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        idx = random.randint(0, len(pil_images) - 1)
        pil_image = pil_images[idx]
        cv_image = cv_images[idx].copy()

        # Create transform that would be very visible if applied
        pil_transform = T.RandomApply([T.ColorJitter(brightness=2.0)], p=0.0)
        cv_transform = cv_transforms.RandomApply(
            [cv_transforms.ColorJitter(brightness=2.0)], p=0.0
        )

        # Apply multiple times to ensure consistency
        for _ in range(10):
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            pil_result = pil_transform(pil_image)

            torch.manual_seed(random_seed)
            random.seed(random_seed)
            cv_result = cv_transform(cv_image)

            # With p=0.0, results should be identical to input
            assert np.array_equal(np.array(pil_image), np.array(pil_result)), (
                "PyTorch should not apply transform with p=0.0"
            )
            assert np.array_equal(cv_image, cv_result), (
                "OpenCV should not apply transform with p=0.0"
            )

    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_random_apply_probability_one(self, test_images, random_seed):
        """Test RandomApply with p=1.0 always applies transforms."""
        pil_images, cv_images = test_images

        # Select random image
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        idx = random.randint(0, len(pil_images) - 1)
        pil_image = pil_images[idx]
        cv_image = cv_images[idx].copy()

        # Create transform that will be visible if applied
        pil_transform = T.RandomApply([T.ColorJitter(brightness=0.5)], p=1.0)
        cv_transform = cv_transforms.RandomApply(
            [cv_transforms.ColorJitter(brightness=0.5)], p=1.0
        )

        # Apply multiple times to ensure consistency
        for _ in range(5):
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            pil_result = pil_transform(pil_image)

            torch.manual_seed(random_seed)
            random.seed(random_seed)
            cv_result = cv_transform(cv_image)

            # With p=1.0, transforms should always be applied
            pil_applied = not np.array_equal(np.array(pil_image), np.array(pil_result))
            cv_applied = not np.array_equal(cv_image, cv_result)

            assert pil_applied, "PyTorch should always apply transform with p=1.0"
            assert cv_applied, "OpenCV should always apply transform with p=1.0"

    def test_random_apply_empty_transforms(self, test_images):
        """Test RandomApply with empty transform list."""
        pil_images, cv_images = test_images
        pil_image = pil_images[0]
        cv_image = cv_images[0].copy()

        # Empty transform list should return unchanged image
        pil_transform = T.RandomApply([], p=0.5)
        cv_transform = cv_transforms.RandomApply([], p=0.5)

        pil_result = pil_transform(pil_image)
        cv_result = cv_transform(cv_image)

        assert np.array_equal(np.array(pil_image), np.array(pil_result)), (
            "PyTorch should return unchanged image for empty transforms"
        )
        assert np.array_equal(cv_image, cv_result), (
            "OpenCV should return unchanged image for empty transforms"
        )

    @pytest.mark.parametrize("p", [0.2, 0.8])
    def test_random_apply_with_compose(self, test_images, p):
        """Test RandomApply works correctly within Compose."""
        pil_images, cv_images = test_images
        pil_image = pil_images[0]
        cv_image = cv_images[0].copy()

        # Create composed transforms
        pil_compose = T.Compose(
            [
                T.RandomApply([T.ColorJitter(brightness=0.3)], p=p),
                T.ToTensor(),
            ]
        )
        cv_compose = cv_transforms.Compose(
            [
                cv_transforms.RandomApply(
                    [cv_transforms.ColorJitter(brightness=0.3)], p=p
                ),
                cv_transforms.ToTensor(),
            ]
        )

        # Test multiple times with different seeds
        for seed in [1, 2, 3]:
            torch.manual_seed(seed)
            random.seed(seed)
            pil_result = pil_compose(pil_image)

            torch.manual_seed(seed)
            random.seed(seed)
            cv_result = cv_compose(cv_image)

            # Should produce tensors of same shape and similar values
            assert pil_result.shape == cv_result.shape, (
                "Composed transforms should produce same shape"
            )
            # Allow for reasonable tolerance due to color jitter differences
            assert torch.allclose(pil_result, cv_result, rtol=0.1, atol=0.1), (
                "Composed results should be similar"
            )

    def test_random_apply_repr(self):
        """Test RandomApply string representation."""
        transform = cv_transforms.RandomApply(
            [
                cv_transforms.ColorJitter(brightness=0.5),
                cv_transforms.RandomHorizontalFlip(),
            ],
            p=0.3,
        )

        repr_str = repr(transform)
        assert "RandomApply" in repr_str
        assert "p=0.3" in repr_str
        assert "ColorJitter" in repr_str
        assert "RandomHorizontalFlip" in repr_str

    @pytest.mark.parametrize("invalid_p", [-0.1, 1.1, 2.0])
    def test_random_apply_invalid_probability(self, invalid_p):
        """Test RandomApply handles invalid probability values gracefully."""
        # This test checks if invalid probabilities are handled
        # Note: PyTorch doesn't validate p values, so we should match that behavior
        transform = cv_transforms.RandomApply(
            [cv_transforms.ColorJitter()], p=invalid_p
        )
        assert transform.p == invalid_p  # Should store the value as-is
