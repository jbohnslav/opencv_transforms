import random

import numpy as np
import pytest
import torch
from torchvision import transforms as T

import opencv_transforms.transforms as cv_transforms


class TestRandomGrayscale:
    @pytest.mark.parametrize("p", [0.0, 1.0])
    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_random_grayscale_probability(self, test_images, p, random_seed):
        """Test RandomGrayscale probability behavior matches torchvision."""
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        pil_images, cv_images = test_images

        # Test with RGB image
        pil_image = pil_images[0]  # RGB image
        cv_image = cv_images[0].copy()

        # Create transforms
        pil_transform = T.RandomGrayscale(p=p)
        cv_transform = cv_transforms.RandomGrayscale(p=p)

        # Apply transforms with same random seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        pil_result = pil_transform(pil_image)

        random.seed(random_seed)
        cv_result = cv_transform(cv_image)

        # Convert PIL result to numpy for comparison
        pil_array = np.array(pil_result)

        # For p=0.0, images should be unchanged
        if p == 0.0:
            assert np.array_equal(np.array(pil_image), pil_array)
            assert np.array_equal(cv_image, cv_result)

        # For p=1.0, images should be grayscale (all channels equal)
        elif p == 1.0:
            # Check if result is grayscale (all channels equal)
            assert len(pil_array.shape) == 3 and pil_array.shape[2] == 3
            assert np.allclose(pil_array[:, :, 0], pil_array[:, :, 1], rtol=1e-5)
            assert np.allclose(pil_array[:, :, 1], pil_array[:, :, 2], rtol=1e-5)

            assert len(cv_result.shape) == 3 and cv_result.shape[2] == 3
            assert np.allclose(cv_result[:, :, 0], cv_result[:, :, 1], rtol=1e-5)
            assert np.allclose(cv_result[:, :, 1], cv_result[:, :, 2], rtol=1e-5)

        # Results should match between implementations
        assert np.allclose(pil_array, cv_result, rtol=1e-3, atol=2)

    def test_random_grayscale_probability_behavior(self, test_images):
        """Test RandomGrayscale probability behavior over multiple trials."""
        pil_images, cv_images = test_images
        cv_image = cv_images[0].copy()

        # Test with p=0.5 over multiple trials
        transform = cv_transforms.RandomGrayscale(p=0.5)

        grayscale_count = 0
        original_count = 0
        trials = 100

        for seed in range(trials):
            random.seed(seed)
            result = transform(cv_image.copy())

            # Check if result is grayscale
            is_grayscale = np.allclose(result[:, :, 0], result[:, :, 1], rtol=1e-5)
            if is_grayscale:
                grayscale_count += 1
            else:
                original_count += 1

        # With p=0.5, expect roughly 50% grayscale (allow 30-70% range)
        grayscale_ratio = grayscale_count / trials
        assert 0.3 <= grayscale_ratio <= 0.7
        assert grayscale_count + original_count == trials

    @pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5])
    def test_random_grayscale_consistency(self, test_images, random_seed):
        """Test RandomGrayscale produces consistent results with same seed."""
        pil_images, cv_images = test_images

        cv_image = cv_images[0].copy()

        transform = cv_transforms.RandomGrayscale(p=0.5)

        # Apply transform twice with same seed
        random.seed(random_seed)
        result1 = transform(cv_image.copy())

        random.seed(random_seed)
        result2 = transform(cv_image.copy())

        assert np.array_equal(result1, result2)

    def test_random_grayscale_repr(self):
        """Test RandomGrayscale __repr__ method."""
        transform = cv_transforms.RandomGrayscale(p=0.3)
        expected = "RandomGrayscale(p=0.3)"
        assert repr(transform) == expected


class TestRandomOrder:
    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_random_order_basic(self, test_images, random_seed):
        """Test RandomOrder applies transforms in random order."""
        random.seed(random_seed)
        pil_images, cv_images = test_images

        cv_image = cv_images[0].copy()

        # Create transforms that have visible effects
        transforms = [
            cv_transforms.RandomHorizontalFlip(p=1.0),
            cv_transforms.RandomVerticalFlip(p=1.0),
        ]

        random_order = cv_transforms.RandomOrder(transforms)

        # Apply transform
        result = random_order(cv_image)

        # Result should be different from original (flips applied)
        assert not np.array_equal(cv_image, result)

        # Should have same shape
        assert cv_image.shape == result.shape

    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_random_order_consistency(self, test_images, random_seed):
        """Test RandomOrder produces consistent results with same seed."""
        pil_images, cv_images = test_images

        cv_image = cv_images[0].copy()

        transforms = [
            cv_transforms.RandomHorizontalFlip(p=1.0),
            cv_transforms.RandomVerticalFlip(p=1.0),
        ]

        random_order = cv_transforms.RandomOrder(transforms)

        # Apply transform twice with same seed
        random.seed(random_seed)
        result1 = random_order(cv_image.copy())

        random.seed(random_seed)
        result2 = random_order(cv_image.copy())

        assert np.array_equal(result1, result2)

    @pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5])
    def test_random_order_different_seeds(self, test_images, random_seed):
        """Test RandomOrder produces different results with different seeds."""
        pil_images, cv_images = test_images

        cv_image = cv_images[0].copy()

        transforms = [
            cv_transforms.RandomHorizontalFlip(p=1.0),
            cv_transforms.RandomVerticalFlip(p=1.0),
            cv_transforms.RandomRotation(degrees=90),
        ]

        random_order = cv_transforms.RandomOrder(transforms)

        # Collect results from different seeds
        results = []
        for seed in range(1, 6):
            random.seed(seed)
            result = random_order(cv_image.copy())
            results.append(result)

        # At least some results should be different
        # (with 3 transforms, there are 6 possible orders)
        unique_results = []
        for result in results:
            is_unique = True
            for unique_result in unique_results:
                if np.array_equal(result, unique_result):
                    is_unique = False
                    break
            if is_unique:
                unique_results.append(result)

        # Should have multiple unique results
        assert len(unique_results) > 1

    def test_random_order_single_transform(self, test_images):
        """Test RandomOrder with single transform."""
        pil_images, cv_images = test_images

        cv_image = cv_images[0].copy()

        # Single transform
        transforms = [cv_transforms.RandomHorizontalFlip(p=1.0)]
        random_order = cv_transforms.RandomOrder(transforms)

        result = random_order(cv_image)

        # Should be horizontally flipped
        expected = cv_transforms.RandomHorizontalFlip(p=1.0)(cv_image)
        assert np.array_equal(result, expected)

    def test_random_order_repr(self):
        """Test RandomOrder __repr__ method."""
        transforms = [
            cv_transforms.RandomHorizontalFlip(),
            cv_transforms.RandomVerticalFlip(),
        ]
        random_order = cv_transforms.RandomOrder(transforms)
        repr_str = repr(random_order)
        assert "RandomOrder" in repr_str
        assert "RandomHorizontalFlip" in repr_str
        assert "RandomVerticalFlip" in repr_str


class TestRandomChoice:
    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_random_choice_basic(self, test_images, random_seed):
        """Test RandomChoice selects one transform randomly."""
        random.seed(random_seed)
        pil_images, cv_images = test_images

        cv_image = cv_images[0].copy()

        # Create transforms with different effects
        transforms = [
            cv_transforms.RandomHorizontalFlip(p=1.0),
            cv_transforms.RandomVerticalFlip(p=1.0),
        ]

        random_choice = cv_transforms.RandomChoice(transforms)

        # Apply transform
        result = random_choice(cv_image)

        # Result should match one of the individual transforms
        hflip_result = transforms[0](cv_image.copy())
        vflip_result = transforms[1](cv_image.copy())

        # Result should match exactly one of the transforms
        matches_hflip = np.array_equal(result, hflip_result)
        matches_vflip = np.array_equal(result, vflip_result)

        assert matches_hflip or matches_vflip
        assert not (matches_hflip and matches_vflip)  # Should match exactly one

    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_random_choice_consistency(self, test_images, random_seed):
        """Test RandomChoice produces consistent results with same seed."""
        pil_images, cv_images = test_images

        cv_image = cv_images[0].copy()

        transforms = [
            cv_transforms.RandomHorizontalFlip(p=1.0),
            cv_transforms.RandomVerticalFlip(p=1.0),
        ]

        random_choice = cv_transforms.RandomChoice(transforms)

        # Apply transform twice with same seed
        random.seed(random_seed)
        result1 = random_choice(cv_image.copy())

        random.seed(random_seed)
        result2 = random_choice(cv_image.copy())

        assert np.array_equal(result1, result2)

    def test_random_choice_distribution(self, test_images):
        """Test RandomChoice selects from all transforms over many trials."""
        pil_images, cv_images = test_images

        cv_image = cv_images[0].copy()

        # Create transforms with easily distinguishable effects
        transforms = [
            cv_transforms.RandomHorizontalFlip(p=1.0),
            cv_transforms.RandomVerticalFlip(p=1.0),
        ]

        random_choice = cv_transforms.RandomChoice(transforms)

        # Apply many times with different seeds
        hflip_count = 0
        vflip_count = 0
        trials = 50

        hflip_expected = transforms[0](cv_image.copy())
        vflip_expected = transforms[1](cv_image.copy())

        for seed in range(trials):
            random.seed(seed)
            result = random_choice(cv_image.copy())

            if np.array_equal(result, hflip_expected):
                hflip_count += 1
            elif np.array_equal(result, vflip_expected):
                vflip_count += 1

        # Both transforms should be selected at least once
        assert hflip_count > 0
        assert vflip_count > 0
        assert hflip_count + vflip_count == trials

    def test_random_choice_single_transform(self, test_images):
        """Test RandomChoice with single transform."""
        pil_images, cv_images = test_images

        cv_image = cv_images[0].copy()

        # Single transform
        transforms = [cv_transforms.RandomHorizontalFlip(p=1.0)]
        random_choice = cv_transforms.RandomChoice(transforms)

        result = random_choice(cv_image)

        # Should be horizontally flipped
        expected = transforms[0](cv_image)
        assert np.array_equal(result, expected)

    def test_random_choice_repr(self):
        """Test RandomChoice __repr__ method."""
        transforms = [
            cv_transforms.RandomHorizontalFlip(),
            cv_transforms.RandomVerticalFlip(),
        ]
        random_choice = cv_transforms.RandomChoice(transforms)
        repr_str = repr(random_choice)
        assert "RandomChoice" in repr_str
        assert "RandomHorizontalFlip" in repr_str
        assert "RandomVerticalFlip" in repr_str
