"""Debug utilities for investigating PIL/OpenCV transform differences.

This module provides tools for comparing and debugging differences between
PIL (torchvision) and OpenCV implementations of image transforms.
"""

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F_pil

from opencv_transforms import functional as F


def compare_contrast_outputs(image, contrast_factor, verbose=True):
    """Compare contrast adjustment between PIL and OpenCV.

    Args:
        image: Input image (PIL Image or numpy array)
        contrast_factor: Factor for contrast adjustment
        verbose: Whether to print detailed comparison results

    Returns:
        dict: Dictionary containing comparison results:
            - equal: Whether outputs are exactly equal
            - num_diff: Number of differing pixels
            - total_pixels: Total number of pixels
            - pil_result: PIL output as numpy array
            - cv_result: OpenCV output as numpy array
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

    Helps understand how PIL calculates contrast adjustments.

    Args:
        test_values: List of pixel values to test
        mean: Mean value used in contrast calculation
        contrast_factor: Contrast adjustment factor
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

    Shows that PIL doesn't always return the original image for contrast=1.0
    due to floating-point precision issues.

    Args:
        image: Input image (PIL Image or numpy array)
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

    Args:
        image: Input image
        contrast_factors: List of contrast factors to test (default: [0.0, 0.5, 1.0, 1.5, 2.0])

    Returns:
        list: List of dictionaries with test results for each contrast factor
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


def compare_rotation_outputs(image, angle, verbose=True):
    """Compare rotation between PIL and OpenCV.

    Args:
        image: Input image (PIL Image or numpy array)
        angle: Rotation angle in degrees
        verbose: Whether to print detailed comparison results

    Returns:
        dict: Dictionary containing comparison results:
            - equal: Whether outputs are exactly equal
            - num_diff: Number of differing pixels
            - total_pixels: Total number of pixels
            - max_diff: Maximum pixel difference
            - mean_diff: Mean pixel difference
            - pil_result: PIL output as numpy array
            - cv_result: OpenCV output as numpy array
    """
    # Handle both PIL and numpy inputs
    if isinstance(image, Image.Image):
        pil_image = image
        cv_image = np.array(image)
    else:
        cv_image = image
        pil_image = Image.fromarray(image)

    # Apply rotation
    pil_rotated = pil_image.rotate(angle)
    cv_rotated = F.rotate(cv_image, angle)

    # Convert to arrays for comparison
    pil_array = np.array(pil_rotated).astype(np.float32)
    cv_array = cv_rotated.astype(np.float32)

    # Calculate differences
    diff = np.abs(pil_array - cv_array)
    are_equal = np.allclose(pil_array, cv_array, atol=0.1)
    num_diff = np.count_nonzero(diff > 0.1)
    max_diff = diff.max()
    mean_diff = diff.mean()

    if verbose:
        print(f"\nRotation angle: {angle}°")
        print(f"Are they close? {are_equal}")
        print(f"Max difference: {max_diff:.2f}")
        print(f"Mean difference: {mean_diff:.4f}")
        print(
            f"Pixels with diff > 0.1: {num_diff} / {pil_array.size} ({num_diff / pil_array.size * 100:.4f}%)"
        )

        if num_diff > 0 and num_diff < 100:
            # Show locations of maximum differences
            max_diff_locations = np.where(diff == max_diff)
            if len(max_diff_locations[0]) > 0:
                y, x = max_diff_locations[0][0], max_diff_locations[1][0]
                if len(max_diff_locations) > 2:  # Color image
                    c = max_diff_locations[2][0]
                    print(
                        f"  Max diff at ({y},{x},{c}): PIL={pil_array[y, x, c]}, CV={cv_array[y, x, c]}"
                    )
                else:  # Grayscale
                    print(
                        f"  Max diff at ({y},{x}): PIL={pil_array[y, x]}, CV={cv_array[y, x]}"
                    )

    return {
        "equal": are_equal,
        "num_diff": num_diff,
        "total_pixels": pil_array.size,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "pil_result": pil_array,
        "cv_result": cv_array,
    }


def analyze_coordinate_system_difference(image, angle, verbose=True):
    """Analyze the coordinate system difference between PIL and OpenCV rotation.

    This function compares the old OpenCV coordinate system (pixel corners)
    with the new fixed coordinate system (pixel centers) to demonstrate
    the improvement from the coordinate system fix.

    Args:
        image: Input image (PIL Image or numpy array)
        angle: Rotation angle in degrees
        verbose: Whether to print detailed analysis

    Returns:
        dict: Dictionary containing analysis results with old vs new coordinate systems
    """
    if isinstance(image, Image.Image):
        pil_image = image
        cv_image = np.array(image)
    else:
        cv_image = image
        pil_image = Image.fromarray(image)

    # PIL rotation (ground truth)
    pil_rotated = pil_image.rotate(angle)
    pil_array = np.array(pil_rotated).astype(np.float32)

    # OpenCV with new coordinate system (current implementation)
    cv_rotated_new = F.rotate(cv_image, angle)

    # OpenCV with old coordinate system (for comparison)
    rows, cols = cv_image.shape[0:2]
    center_old = (cols / 2, rows / 2)  # Old method
    M_old = cv2.getRotationMatrix2D(center_old, angle, 1)

    if len(cv_image.shape) == 2 or (
        len(cv_image.shape) == 3 and cv_image.shape[2] == 1
    ):
        cv_rotated_old = cv2.warpAffine(
            cv_image,
            M_old,
            (cols, rows),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )[:, :, np.newaxis]
    else:
        cv_rotated_old = cv2.warpAffine(
            cv_image,
            M_old,
            (cols, rows),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # Calculate differences
    diff_old = np.abs(pil_array - cv_rotated_old.astype(np.float32))
    diff_new = np.abs(pil_array - cv_rotated_new.astype(np.float32))

    num_diff_old = np.count_nonzero(diff_old > 0.1)
    num_diff_new = np.count_nonzero(diff_new > 0.1)

    max_diff_old = diff_old.max()
    max_diff_new = diff_new.max()

    mean_diff_old = diff_old.mean()
    mean_diff_new = diff_new.mean()

    if verbose:
        print(f"\nCoordinate System Analysis for {angle}° rotation:")
        print(
            f"{'Method':<20} {'Max Diff':<10} {'Mean Diff':<12} {'Diff Pixels':<15} {'Percentage':<10}"
        )
        print("-" * 70)

        old_percent = (num_diff_old / pil_array.size) * 100
        new_percent = (num_diff_new / pil_array.size) * 100

        print(
            f"{'Old coordinates':<20} {max_diff_old:<10.1f} {mean_diff_old:<12.4f} {num_diff_old:<15d} {old_percent:<10.2f}%"
        )
        print(
            f"{'New coordinates':<20} {max_diff_new:<10.1f} {mean_diff_new:<12.4f} {num_diff_new:<15d} {new_percent:<10.2f}%"
        )

        improvement_pixels = num_diff_old - num_diff_new
        improvement_max = max_diff_old - max_diff_new

        print("\nImprovement:")
        print(
            f"  Pixel differences reduced by: {improvement_pixels:,} ({improvement_pixels / num_diff_old * 100:.1f}%)"
        )
        print(f"  Max difference reduced by: {improvement_max:.1f}")

    return {
        "old_system": {
            "max_diff": float(max_diff_old),
            "mean_diff": float(mean_diff_old),
            "num_diff": num_diff_old,
            "result": cv_rotated_old,
        },
        "new_system": {
            "max_diff": float(max_diff_new),
            "mean_diff": float(mean_diff_new),
            "num_diff": num_diff_new,
            "result": cv_rotated_new,
        },
        "pil_result": pil_array,
        "improvement": {
            "pixel_reduction": improvement_pixels,
            "max_diff_reduction": float(improvement_max),
        },
    }


def test_rotation_angles(image, angles=None, verbose=True):  # noqa: PT028
    """Test rotation accuracy across multiple angles.

    Args:
        image: Input image (PIL Image or numpy array)
        angles: List of angles to test (default: [10, 30, 45, 90, 180])
        verbose: Whether to print detailed results

    Returns:
        list: List of comparison results for each angle
    """
    if angles is None:
        angles = [10, 30, 45, 90, 180]

    results = []

    if verbose:
        print("Rotation Accuracy Test")
        print("=" * 50)
        print(
            f"{'Angle':<8} {'Max Diff':<10} {'Mean Diff':<12} {'Diff Pixels':<15} {'Pass':<8}"
        )
        print("-" * 50)

    for angle in angles:
        result = compare_rotation_outputs(image, angle, verbose=False)
        result["angle"] = angle

        # Define pass/fail criteria (matching test tolerances)
        passes = result["max_diff"] <= 220.0  # Current rotation tolerance
        result["passes"] = passes

        if verbose:
            status = "PASS" if passes else "FAIL"
            print(
                f"{angle:<8}° {result['max_diff']:<10.1f} {result['mean_diff']:<12.4f} {result['num_diff']:<15d} {status:<8}"
            )

        results.append(result)

    if verbose:
        passed = sum(1 for r in results if r["passes"])
        print(f"\nResults: {passed}/{len(results)} angles passed")

    return results
