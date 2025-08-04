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

## Project Structure

- `opencv_transforms/` - Main package code
- `tests/` - Unit tests
- `TEST_PLAN.md` - Documentation of missing tests
- `UPDATE.md` - Documentation of missing transforms compared to torchvision