from typing import Union

import numpy as np
from PIL.Image import Image as PIL_image  # for typing

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def L1(pil: Union[PIL_image, np.ndarray], np_image: np.ndarray) -> float:
    return np.abs(np.asarray(pil) - np_image).mean()


def assert_transforms_close(
    pil_output: Union[PIL_image, np.ndarray],
    cv_output: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-3,
    pixel_atol: float = 1.0,
    backend: str = "auto",
) -> None:
    """Assert that two transform outputs are close enough.

    Args:
        pil_output: Output from torchvision transform (PIL Image or numpy array)
        cv_output: Output from opencv_transforms
        rtol: Relative tolerance for np.allclose
        atol: Absolute tolerance for np.allclose (normalized to 0-1 range)
        pixel_atol: Absolute tolerance in pixel values (0-255 range)
        backend: "numpy", "torch", or "auto" (auto uses torch if available)
    """
    pil_array = np.asarray(pil_output).astype(np.float32)
    cv_array = cv_output.astype(np.float32)

    # Check if shapes match
    assert pil_array.shape == cv_array.shape, (
        f"Shape mismatch: PIL {pil_array.shape} vs CV {cv_array.shape}"
    )

    # Decide backend
    use_torch = backend == "torch" or (backend == "auto" and TORCH_AVAILABLE)

    if use_torch:
        # Convert to torch tensors
        pil_tensor = torch.from_numpy(pil_array)
        cv_tensor = torch.from_numpy(cv_array)

        # Check with torch.allclose (operates on raw pixel values)
        if not torch.allclose(pil_tensor, cv_tensor, rtol=rtol, atol=pixel_atol):
            pixel_diff = torch.abs(pil_tensor - cv_tensor)
            max_diff = pixel_diff.max().item()
            mean_diff = pixel_diff.mean().item()

            raise AssertionError(
                f"Transform outputs differ too much (torch.allclose):\n"
                f"  Max pixel difference: {max_diff:.2f} (threshold: {pixel_atol})\n"
                f"  Mean pixel difference: {mean_diff:.2f}\n"
                f"  Shape: {pil_array.shape}"
            )
    else:
        # Normalize to 0-1 range for relative comparison
        pil_norm = pil_array / 255.0
        cv_norm = cv_array / 255.0

        # Check with both relative and absolute tolerance
        if not np.allclose(pil_norm, cv_norm, rtol=rtol, atol=atol):
            # If normalized check fails, check absolute pixel difference
            pixel_diff = np.abs(pil_array - cv_array)
            max_diff = pixel_diff.max()
            mean_diff = pixel_diff.mean()

            assert max_diff <= pixel_atol, (
                f"Transform outputs differ too much (np.allclose):\n"
                f"  Max pixel difference: {max_diff:.2f} (threshold: {pixel_atol})\n"
                f"  Mean pixel difference: {mean_diff:.2f}\n"
                f"  Shape: {pil_array.shape}"
            )
