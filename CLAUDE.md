# opencv_transforms Project Guidelines

## Development Environment

**ALWAYS use `uv` for Python package management** - Never use pip, pip3, or create venv manually.

### Setting up the environment:
```bash
# Create virtual environment and install all dependencies (including dev)
uv sync

# Activate it
source .venv/bin/activate
```

**Note**: `uv sync` automatically installs all dependencies including dev dependencies. Never use `uv pip install`.

## Pre-commit Hooks

**NEVER commit without pre-commit hooks!** This project uses pre-commit hooks for code quality.

### Pre-commit workflow:
1. Install pre-commit hooks: `pre-commit install`
2. Run on all files: `pre-commit run --all-files`
3. The hooks will automatically run on `git commit`

### Current hooks:
- `ruff` for linting
- `ruff-format` for formatting

## Git Workflow

1. Always run pre-commit hooks before committing
2. Never use `--no-verify` flag
3. Never bypass pre-commit hooks

## Testing Philosophy

**PyTorch/torchvision is the ground truth** - All OpenCV transforms must match PyTorch transforms across:
- Different image sizes
- Different image types
- Different parameter values

## Documentation Standards

**Document PIL/OpenCV differences in docstrings** - When implementing transforms:
- If there are known precision differences between PIL and OpenCV, document them in the function's docstring
- Include the magnitude of differences (e.g., "±1 pixel value for <0.01% of pixels")
- Explain the root cause if known (e.g., "PIL has floating-point precision issues")
- Add implementation comments explaining any workarounds to match PIL behavior

Example:
```python
def some_transform(img, param):
    """Transform description.
    
    Note:
        Small differences (±1 pixel value) may occur for a tiny fraction of pixels
        (<0.01%) due to floating-point precision differences between PIL and OpenCV.
    """
```

## Project Structure

- `opencv_transforms/` - Main package code
- `tests/` - Unit tests
- `debug/` - Debug utilities for investigating PIL/OpenCV differences
  - `debug_utils.py` - Consolidated debugging functions
  - Various investigation scripts from debugging sessions
- `TEST_PLAN.md` - Documentation of missing tests
- `UPDATE.md` - Documentation of missing transforms compared to torchvision

## Debugging Transform Differences

Use the debug utilities when investigating transform differences:

```python
from debug.debug_utils import compare_contrast_outputs, test_beans_dataset_image

# Debug specific transform
result = compare_contrast_outputs(image, contrast_factor=0.5)

# Test with the actual test fixture image
test_beans_dataset_image()
```

See `debug/README.md` for documentation on available debugging tools.