"""Consolidated debug utilities from various debugging sessions.

This module combines the useful debugging functions developed while
investigating PIL/OpenCV transform differences.
"""

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F


def compare_contrast_outputs(image, contrast_factor, verbose=True):
    """Compare contrast adjustment between PIL and OpenCV.

    This function was developed from debug_contrast_real.py and other scripts.
    """
    # Handle both PIL and numpy inputs
    if isinstance(image, Image.Image):
        pil_image = image
        cv_image = np.array(image)
    else:
        cv_image = image
        if image.ndim == 2:
            pil_image = Image.fromarray(image, mode="L")
        else:
            pil_image = Image.fromarray(image)

    # For grayscale
    if cv_image.ndim == 3:
        cv_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        pil_gray = pil_image.convert("L")
    else:
        cv_gray = cv_image
        pil_gray = pil_image

    # Apply transforms
    pil_enhanced = F_pil.adjust_contrast(pil_gray, contrast_factor)
    np_enhanced = F.adjust_contrast(cv_gray, contrast_factor)

    pil_array = np.array(pil_enhanced)
    cv_squeezed = np_enhanced.squeeze()

    # Analysis
    are_equal = np.array_equal(pil_array, cv_squeezed)
    diff_mask = pil_array != cv_squeezed
    num_diff = np.sum(diff_mask)

    if verbose:
        print(f"\nContrast factor: {contrast_factor}")
        print(f"Are they equal? {are_equal}")
        if not are_equal:
            print(
                f"Number of different pixels: {num_diff} / {pil_array.size} ({num_diff / pil_array.size * 100:.4f}%)"
            )

            # Show first few differences
            if num_diff > 0 and num_diff < 100:
                indices = np.where(diff_mask)
                for i in range(min(5, len(indices[0]))):
                    r, c = indices[0][i], indices[1][i]
                    print(
                        f"  ({r},{c}): PIL={pil_array[r, c]}, CV={cv_squeezed[r, c]}, diff={int(pil_array[r, c]) - int(cv_squeezed[r, c])}"
                    )

    return {
        "equal": are_equal,
        "num_diff": num_diff,
        "total_pixels": pil_array.size,
        "pil_result": pil_array,
        "cv_result": cv_squeezed,
    }


def debug_contrast_formula(test_values, mean, contrast_factor):
    """Debug the exact contrast formula calculations.

    From debug_exact_computation.py - helps understand how PIL calculates contrast.
    """
    print(f"\nMean: {mean}")
    print(f"Contrast factor: {contrast_factor}")

    # Test different formulas
    print("\n--- Testing formulas ---")
    for val in test_values:
        # Standard contrast formula
        result = (val - mean) * contrast_factor + mean
        result_rounded = int(result + 0.5)

        print(f"Value {val}:")
        print(
            f"  Formula: ({val} - {mean:.2f}) * {contrast_factor} + {mean:.2f} = {result:.2f}"
        )
        print(f"  Rounded: {result_rounded}")


def analyze_pil_precision_issue(image):
    """Analyze PIL's precision issues with contrast=1.0.

    From debug_test_setup.py - shows that PIL doesn't always return
    the original image for contrast=1.0.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_image = Image.fromarray(image, mode="L")
        else:
            pil_image = Image.fromarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            pil_image = pil_image.convert("L")

    # Apply contrast=1.0 (should be identity)
    pil_result = F_pil.adjust_contrast(pil_image, 1.0)
    pil_array = np.array(pil_result)

    # Check if it's actually identity
    original = np.array(image) if isinstance(image, Image.Image) else image

    is_identity = np.array_equal(original, pil_array)
    diff_count = np.sum(original != pil_array)

    print("\nPIL contrast=1.0 precision check:")
    print(f"Returns original image? {is_identity}")
    if not is_identity:
        print(
            f"Number of changed pixels: {diff_count} ({diff_count / original.size * 100:.4f}%)"
        )

        # Show some examples
        indices = np.where(original != pil_array)
        for i in range(min(5, len(indices[0]))):
            r, c = indices[0][i], indices[1][i]
            print(f"  ({r},{c}): {original[r, c]} -> {pil_array[r, c]}")


def create_contrast_test_summary(image, contrast_factors=None):
    """Create a summary of contrast test results across multiple factors.

    Consolidated from test_contrast_directly.py and final_debug.py.
    """
    if contrast_factors is None:
        contrast_factors = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = []

    for cf in contrast_factors:
        result = compare_contrast_outputs(image, cf, verbose=False)
        result["contrast_factor"] = cf
        results.append(result)

    # Summary
    print("\nContrast Test Summary")
    print("-" * 50)
    print(f"{'Factor':<10} {'Equal':<10} {'Diff Pixels':<15} {'Percent':<10}")
    print("-" * 50)

    for r in results:
        percent = (
            (r["num_diff"] / r["total_pixels"] * 100) if r["total_pixels"] > 0 else 0
        )
        print(
            f"{r['contrast_factor']:<10.1f} {r['equal']!s:<10} {r['num_diff']:<15d} {percent:<10.4f}%"
        )

    return results


# Specific test from the debugging session
def test_beans_dataset_image():
    """Test with the actual image that was failing in tests."""
    from datasets import load_dataset  # noqa: PLC0415

    print("Loading beans dataset test image...")
    train_dataset = load_dataset(
        "beans", split="train", streaming=True, cache_dir="tests/.cache/"
    )

    samples = []
    for i, sample in enumerate(train_dataset):
        if i >= 50:
            break
        samples.append(sample["image"])

    # Use first image (same as single_test_image fixture)
    pil_image = samples[0]

    print("\nTesting with beans dataset image (same as test fixture)")
    create_contrast_test_summary(pil_image)

    # Check precision issue
    analyze_pil_precision_issue(pil_image)


if __name__ == "__main__":
    # Run test with beans dataset image
    test_beans_dataset_image()
