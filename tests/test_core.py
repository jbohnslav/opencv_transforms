import random

import numpy as np
import pytest
import torch
from torchvision import transforms as T

import opencv_transforms.transforms as cv_transforms


class TestCompose:
    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_compose_single_transform(self, test_images, random_seed):
        """Test Compose with single transform matches torchvision."""
        random.seed(random_seed)
        pil_images, cv_images = test_images

        # Select random image
        idx = random.randint(0, len(pil_images) - 1)
        pil_image = pil_images[idx]
        cv_image = cv_images[idx].copy()

        # Single transform composition
        pil_compose = T.Compose([T.ToTensor()])
        cv_compose = cv_transforms.Compose([cv_transforms.ToTensor()])

        pil_result = pil_compose(pil_image)
        cv_result = cv_compose(cv_image)

        assert torch.allclose(pil_result, cv_result, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_compose_multiple_transforms(self, test_images, random_seed):
        """Test Compose with multiple transforms matches torchvision."""
        random.seed(random_seed)
        pil_images, cv_images = test_images

        # Select random image
        idx = random.randint(0, len(pil_images) - 1)
        pil_image = pil_images[idx]
        cv_image = cv_images[idx].copy()

        # Multiple transform composition
        pil_compose = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        cv_compose = cv_transforms.Compose(
            [
                cv_transforms.ToTensor(),
                cv_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        pil_result = pil_compose(pil_image)
        cv_result = cv_compose(cv_image)

        assert torch.allclose(pil_result, cv_result, rtol=1e-5, atol=1e-6)

    def test_compose_repr(self):
        """Test Compose string representation."""
        transforms = [cv_transforms.ToTensor(), cv_transforms.Normalize([0.5], [0.5])]
        compose = cv_transforms.Compose(transforms)

        repr_str = repr(compose)
        assert "Compose(" in repr_str
        assert "ToTensor()" in repr_str
        assert "Normalize(" in repr_str

    def test_compose_empty_transforms(self):
        """Test Compose with empty transform list."""
        pil_image = torch.randn(3, 224, 224)
        compose = cv_transforms.Compose([])

        result = compose(pil_image)
        assert torch.equal(result, pil_image)


class TestToTensor:
    @pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5])
    def test_to_tensor_pil_equivalence(self, test_images, random_seed):
        """Test ToTensor matches torchvision when converting PIL to numpy first."""
        random.seed(random_seed)
        pil_images, cv_images = test_images

        # Select random image
        idx = random.randint(0, len(pil_images) - 1)
        pil_image = pil_images[idx]
        cv_image = cv_images[idx]

        pil_transform = T.ToTensor()
        cv_transform = cv_transforms.ToTensor()

        pil_result = pil_transform(pil_image)
        cv_result = cv_transform(cv_image)

        assert torch.allclose(pil_result, cv_result, rtol=1e-7, atol=1e-8)

    @pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5])
    def test_to_tensor_numpy_equivalence(self, test_images, random_seed):
        """Test ToTensor matches torchvision on numpy arrays."""
        random.seed(random_seed)
        _, cv_images = test_images

        # Select random image
        idx = random.randint(0, len(cv_images) - 1)
        cv_image = cv_images[idx]

        pil_transform = T.ToTensor()
        cv_transform = cv_transforms.ToTensor()

        pil_result = pil_transform(cv_image)
        cv_result = cv_transform(cv_image)

        assert torch.allclose(pil_result, cv_result, rtol=1e-7, atol=1e-8)

    def test_to_tensor_range_conversion(self, single_test_image):
        """Test ToTensor properly converts from [0, 255] to [0.0, 1.0]."""
        _, cv_image = single_test_image

        cv_transform = cv_transforms.ToTensor()

        # Test with numpy array
        np_result = cv_transform(cv_image)
        assert np_result.min() >= 0.0
        assert np_result.max() <= 1.0
        assert np_result.dtype == torch.float32

    def test_to_tensor_channel_order(self, single_test_image):
        """Test ToTensor properly converts HWC to CHW format."""
        _, cv_image = single_test_image

        cv_transform = cv_transforms.ToTensor()
        result = cv_transform(cv_image)

        # Original image is HWC, result should be CHW
        assert len(result.shape) == 3
        assert result.shape[0] == cv_image.shape[2]  # Channels
        assert result.shape[1] == cv_image.shape[0]  # Height
        assert result.shape[2] == cv_image.shape[1]  # Width

    def test_to_tensor_grayscale(self):
        """Test ToTensor works with grayscale images."""
        # Create a grayscale image with an extra dimension for channels
        gray_image = np.random.randint(0, 255, (224, 224, 1), dtype=np.uint8)

        cv_transform = cv_transforms.ToTensor()
        result = cv_transform(gray_image)

        assert result.shape == (1, 224, 224)  # CHW format with 1 channel
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_to_tensor_repr(self):
        """Test ToTensor string representation."""
        transform = cv_transforms.ToTensor()
        assert repr(transform) == "ToTensor()"


class TestNormalize:
    @pytest.mark.parametrize(
        "mean,std",
        [
            ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
            ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Simple normalization
            ([0.0], [1.0]),  # Grayscale
        ],
    )
    def test_normalize_equivalence(self, single_test_image, mean, std):
        """Test Normalize matches torchvision."""
        pil_image, _ = single_test_image

        # Convert to tensor first
        tensor_image = T.ToTensor()(pil_image)

        pil_transform = T.Normalize(mean=mean, std=std)
        cv_transform = cv_transforms.Normalize(mean=mean, std=std)

        pil_result = pil_transform(tensor_image.clone())
        cv_result = cv_transform(tensor_image.clone())

        assert torch.allclose(pil_result, cv_result, rtol=1e-6, atol=1e-7)

    def test_normalize_inplace_behavior(self, single_test_image):
        """Test Normalize acts in-place."""
        pil_image, _ = single_test_image
        tensor_image = T.ToTensor()(pil_image)
        original_tensor = tensor_image.clone()

        cv_transform = cv_transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        result = cv_transform(tensor_image)

        # Should be same object (in-place)
        assert result is tensor_image
        # Should be different from original
        assert not torch.equal(tensor_image, original_tensor)

    def test_normalize_grayscale_tensor(self):
        """Test Normalize works with grayscale tensors."""
        # Create grayscale tensor
        gray_tensor = torch.rand(1, 224, 224)

        cv_transform = cv_transforms.Normalize(mean=[0.5], std=[0.5])
        result = cv_transform(gray_tensor.clone())

        expected = (gray_tensor - 0.5) / 0.5
        assert torch.allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_normalize_formula(self):
        """Test Normalize applies correct formula: (input - mean) / std."""
        # Create test tensor with known values
        test_tensor = torch.ones(3, 2, 2) * 0.5  # All values are 0.5
        mean = [0.25, 0.5, 0.75]
        std = [0.1, 0.2, 0.3]

        cv_transform = cv_transforms.Normalize(mean=mean, std=std)
        result = cv_transform(test_tensor.clone())

        # Expected result: (0.5 - mean) / std for each channel
        expected = torch.zeros_like(test_tensor)
        expected[0] = (0.5 - 0.25) / 0.1  # 2.5
        expected[1] = (0.5 - 0.5) / 0.2  # 0.0
        expected[2] = (0.5 - 0.75) / 0.3  # -0.833...

        assert torch.allclose(result, expected, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("random_seed", [1, 2, 3])
    def test_normalize_with_compose(self, test_images, random_seed):
        """Test Normalize works correctly in composition."""
        random.seed(random_seed)
        pil_images, cv_images = test_images

        # Select random image
        idx = random.randint(0, len(pil_images) - 1)
        pil_image = pil_images[idx]
        cv_image = cv_images[idx].copy()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Full pipeline comparison
        pil_compose = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
        cv_compose = cv_transforms.Compose(
            [cv_transforms.ToTensor(), cv_transforms.Normalize(mean=mean, std=std)]
        )

        pil_result = pil_compose(pil_image)
        cv_result = cv_compose(cv_image)

        assert torch.allclose(pil_result, cv_result, rtol=1e-5, atol=1e-6)
