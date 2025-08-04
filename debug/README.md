# Debug Directory

This directory contains debugging utilities and scripts developed while investigating differences between PIL/torchvision and OpenCV transform implementations.

## Main Debug Utilities

### `debug_utils.py`
Consolidated debugging functions from various investigation sessions:

- `compare_contrast_outputs()` - Compare PIL and OpenCV contrast results
- `debug_contrast_formula()` - Debug exact formula calculations
- `analyze_pil_precision_issue()` - Analyze PIL's precision bugs
- `create_contrast_test_summary()` - Test multiple contrast factors
- `test_beans_dataset_image()` - Test with the actual failing test image

Example usage:
```python
from debug.debug_utils import compare_contrast_outputs, test_beans_dataset_image

# Test specific image
import numpy as np
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
result = compare_contrast_outputs(image, contrast_factor=0.5)

# Or test with the actual test fixture image
test_beans_dataset_image()
```

## Individual Debug Scripts

These scripts were created during the investigation of contrast transform failures:

- `debug_contrast*.py` - Various approaches to debugging contrast
- `debug_pil*.py` - Understanding PIL's implementation details
- `debug_blend_formula.py` - Testing different blend formulas
- `final_debug.py` - Final debugging session results

## Key Findings

1. PIL has precision issues where `contrast_factor=1.0` doesn't always return the original image
2. Differences are typically ±1 pixel value for <0.01% of pixels
3. PIL uses `int(value + 0.5)` for rounding
4. The mean value calculation is critical for matching PIL behavior