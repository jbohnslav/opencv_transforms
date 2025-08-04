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


class TestLinearTransformation:
    def test_linear_transformation_init_valid_square_matrix(self):
        """Test LinearTransformation initialization with valid square matrix."""
        # 3x3 identity matrix for 1x1x3 image
        matrix = torch.eye(3)
        transform = cv_transforms.LinearTransformation(matrix)
        assert torch.equal(transform.transformation_matrix, matrix)

    def test_linear_transformation_init_non_square_matrix_error(self):
        """Test LinearTransformation raises error for non-square matrix."""
        # Non-square matrix
        matrix = torch.rand(3, 4)
        with pytest.raises(ValueError, match="transformation_matrix should be square"):
            cv_transforms.LinearTransformation(matrix)

    def test_linear_transformation_identity_transform(self):
        """Test LinearTransformation with identity matrix produces no change."""
        # Create test tensor
        tensor = torch.rand(3, 4, 4)  # 3x4x4 = 48 elements

        # Create identity transformation matrix
        D = 3 * 4 * 4  # 48
        identity_matrix = torch.eye(D)
        transform = cv_transforms.LinearTransformation(identity_matrix)

        result = transform(tensor)

        # Result should be identical to input for identity transform
        assert torch.allclose(result, tensor, rtol=1e-6, atol=1e-7)
        assert result.shape == tensor.shape

    def test_linear_transformation_simple_scaling(self):
        """Test LinearTransformation with simple scaling matrix."""
        # Create test tensor with known values
        tensor = torch.ones(2, 2, 2)  # 2x2x2 = 8 elements, all ones

        # Create scaling matrix (multiply by 2)
        D = 2 * 2 * 2  # 8
        scaling_matrix = torch.eye(D) * 2.0
        transform = cv_transforms.LinearTransformation(scaling_matrix)

        result = transform(tensor)
        expected = tensor * 2.0

        assert torch.allclose(result, expected, rtol=1e-6, atol=1e-7)
        assert result.shape == tensor.shape

    def test_linear_transformation_shape_preservation(self):
        """Test LinearTransformation preserves tensor shape."""
        shapes = [(1, 1, 1), (3, 32, 32), (1, 28, 28), (3, 64, 64)]

        for shape in shapes:
            tensor = torch.rand(shape)
            D = shape[0] * shape[1] * shape[2]
            matrix = torch.eye(D)
            transform = cv_transforms.LinearTransformation(matrix)

            result = transform(tensor)
            assert result.shape == tensor.shape

    def test_linear_transformation_incompatible_tensor_error(self):
        """Test LinearTransformation raises error for incompatible tensor size."""
        # Create transformation matrix for 3x3x3 tensor (27 elements)
        matrix = torch.eye(27)
        transform = cv_transforms.LinearTransformation(matrix)

        # Try to apply to tensor with different total size
        tensor = torch.rand(2, 2, 2)  # 8 elements, incompatible with 27

        with pytest.raises(
            ValueError, match="tensor and transformation matrix have incompatible shape"
        ):
            transform(tensor)

    def test_linear_transformation_grayscale(self):
        """Test LinearTransformation works with grayscale images."""
        # Grayscale tensor
        tensor = torch.rand(1, 28, 28)
        D = 1 * 28 * 28  # 784

        # Identity transformation
        matrix = torch.eye(D)
        transform = cv_transforms.LinearTransformation(matrix)

        result = transform(tensor)
        assert torch.allclose(result, tensor, rtol=1e-6, atol=1e-7)
        assert result.shape == (1, 28, 28)

    def test_linear_transformation_different_dtypes(self):
        """Test LinearTransformation works with different tensor dtypes."""
        dtypes = [torch.float32, torch.float64]

        for dtype in dtypes:
            tensor = torch.rand(2, 3, 3, dtype=dtype)
            D = 2 * 3 * 3  # 18
            matrix = torch.eye(D, dtype=dtype)
            transform = cv_transforms.LinearTransformation(matrix)

            result = transform(tensor)
            assert result.dtype == dtype
            assert result.shape == tensor.shape

    def test_linear_transformation_single_pixel(self):
        """Test LinearTransformation with single pixel images."""
        # Single pixel RGB image
        tensor = torch.rand(3, 1, 1)
        matrix = torch.eye(3)
        transform = cv_transforms.LinearTransformation(matrix)

        result = transform(tensor)
        assert torch.allclose(result, tensor, rtol=1e-6, atol=1e-7)
        assert result.shape == (3, 1, 1)

    def test_linear_transformation_zero_matrix(self):
        """Test LinearTransformation with zero matrix."""
        tensor = torch.rand(2, 2, 2)
        D = 2 * 2 * 2  # 8
        zero_matrix = torch.zeros(D, D)
        transform = cv_transforms.LinearTransformation(zero_matrix)

        result = transform(tensor)
        expected = torch.zeros_like(tensor)

        assert torch.allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_linear_transformation_random_matrix(self):
        """Test LinearTransformation with random transformation matrix."""
        tensor = torch.rand(3, 4, 4)
        D = 3 * 4 * 4  # 48

        # Create random orthogonal matrix for stable transformation
        random_matrix = torch.randn(D, D)
        U, _, V = torch.svd(random_matrix)
        orthogonal_matrix = torch.mm(U, V.t())

        transform = cv_transforms.LinearTransformation(orthogonal_matrix)
        result = transform(tensor)

        # Should preserve shape but change values
        assert result.shape == tensor.shape
        assert not torch.allclose(result, tensor, rtol=1e-3, atol=1e-3)

    def test_linear_transformation_mathematical_correctness(self):
        """Test LinearTransformation applies correct mathematical operation."""
        # Create simple test case we can verify manually
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # 1x2x2, flattened: [1,2,3,4]

        # Create transformation matrix that reverses the order
        # Flattened tensor: [1, 2, 3, 4] -> [4, 3, 2, 1]
        D = 4
        reverse_matrix = torch.zeros(D, D)
        for i in range(D):
            reverse_matrix[i, D - 1 - i] = 1.0

        transform = cv_transforms.LinearTransformation(reverse_matrix)
        result = transform(tensor)

        # Expected: [[4, 3], [2, 1]] reshaped to 1x2x2
        expected = torch.tensor([[[4.0, 3.0], [2.0, 1.0]]])
        assert torch.allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_linear_transformation_repr(self):
        """Test LinearTransformation string representation."""
        matrix = torch.eye(2)
        transform = cv_transforms.LinearTransformation(matrix)

        repr_str = repr(transform)
        assert "LinearTransformation" in repr_str
        assert "[[1.0, 0.0], [0.0, 1.0]]" in repr_str

    def test_linear_transformation_large_tensor(self):
        """Test LinearTransformation works with larger tensors."""
        # Test with a reasonably sized tensor
        tensor = torch.rand(3, 8, 8)  # 192 elements
        D = 3 * 8 * 8

        # Use identity for predictable results
        matrix = torch.eye(D)
        transform = cv_transforms.LinearTransformation(matrix)

        result = transform(tensor)
        assert torch.allclose(result, tensor, rtol=1e-6, atol=1e-7)
        assert result.shape == tensor.shape

    def test_linear_transformation_batch_compatibility(self):
        """Test LinearTransformation is designed for single tensor (not batched)."""
        # LinearTransformation expects single tensor, not batch
        # This test verifies the expected behavior with 3D tensors only
        tensor = torch.rand(3, 4, 4)
        D = 3 * 4 * 4
        matrix = torch.eye(D)
        transform = cv_transforms.LinearTransformation(matrix)

        result = transform(tensor)
        assert result.shape == tensor.shape

        # Verify it doesn't work with 4D batch tensors
        batch_tensor = torch.rand(2, 3, 4, 4)  # Batch of 2
        with pytest.raises(ValueError):
            transform(batch_tensor)
