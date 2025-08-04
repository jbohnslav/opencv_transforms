"""Debug utilities for opencv_transforms.

This module provides debugging and comparison tools for investigating
differences between PIL (torchvision) and OpenCV implementations.

Basic usage:
    from opencv_transforms.debug import utils
    result = utils.compare_contrast_outputs(image, contrast_factor=0.5)

Visualization (requires matplotlib):
    try:
        from opencv_transforms.debug.visualization import create_comparison_figure
        create_comparison_figure(original, pil_result, cv_result)
    except ImportError:
        print("matplotlib required for visualization")

Dataset testing (requires datasets library):
    try:
        from opencv_transforms.debug.dataset_utils import test_with_dataset_image
        results = test_with_dataset_image("beans")
    except ImportError:
        print("datasets library required for dataset testing")
"""

# Import core utilities (no extra dependencies)
from . import utils

# Optional imports with graceful degradation
try:
    from . import visualization
except ImportError:
    visualization = None

try:
    from . import dataset_utils
except ImportError:
    dataset_utils = None

__all__ = ["utils"]

# Add optional modules to __all__ if available
if visualization is not None:
    __all__.append("visualization")
if dataset_utils is not None:
    __all__.append("dataset_utils")
