"""Debug utilities for analyzing rotation transform differences between PIL and OpenCV."""

import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as pil_transforms
from opencv_transforms import transforms
from opencv_transforms import functional as F
from datasets import load_dataset


def analyze_rotation_differences(angle=10, use_test_image=False, verbose=True):
    """Analyze differences between PIL and OpenCV rotation implementations.

    Args:
        angle: Rotation angle in degrees (or max angle for RandomRotation)
        use_test_image: If True, use the beans dataset image; if False, use synthetic image
        verbose: If True, print detailed analysis

    Returns:
        dict: Analysis results including max_diff, mean_diff, percentage of pixels over threshold
    """
    if use_test_image:
        # Load actual test image from beans dataset
        if verbose:
            print("Loading beans dataset image...")
        dataset = load_dataset("beans", split="train", streaming=True)
        sample = next(iter(dataset))
        pil_image = sample["image"]
        cv_image = np.array(pil_image)
    else:
        # Create synthetic test image with clear features
        cv_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv_image[40:60, 40:60] = [255, 0, 0]  # Red square in center
        cv_image[20:30, 20:30] = [0, 255, 0]  # Green square top-left
        cv_image[20:30, 70:80] = [0, 0, 255]  # Blue square top-right
        pil_image = Image.fromarray(cv_image)

    # Test with fixed angle
    if verbose:
        print(f"\nTesting fixed angle rotation ({angle}°)...")

    # PIL rotation
    pil_rotated = pil_image.rotate(angle)
    pil_array = np.array(pil_rotated)

    # OpenCV rotation
    cv_rotated = F.rotate(cv_image, angle)

    # Analyze differences
    diff = np.abs(pil_array.astype(float) - cv_rotated.astype(float))
    max_diff = diff.max()
    mean_diff = diff.mean()
    pixels_over_120 = np.sum(diff > 120)
    percent_over_120 = 100 * pixels_over_120 / diff.size

    if verbose:
        print(f"Fixed angle {angle}°:")
        print(f"  Max difference: {max_diff}")
        print(f"  Mean difference: {mean_diff:.2f}")
        print(f"  Pixels > 120: {pixels_over_120} ({percent_over_120:.2f}%)")

    # If max diff is 255, find where it occurs
    if max_diff == 255 and verbose:
        max_coords = np.where(diff == 255)
        if len(max_coords[0]) > 0:
            y, x, c = max_coords[0][0], max_coords[1][0], max_coords[2][0]
            print(f"  Max diff at pixel ({x}, {y}, channel {c}):")
            print(f"    PIL value: {pil_array[y, x, c]}")
            print(f"    CV value: {cv_rotated[y, x, c]}")

    results = {
        "fixed_angle": {
            "angle": angle,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "pixels_over_120": pixels_over_120,
            "percent_over_120": percent_over_120,
        }
    }

    return results


def test_random_rotation_sync(max_angle=10, verbose=True):
    """Test synchronization of RandomRotation between PIL and OpenCV.

    Args:
        max_angle: Maximum rotation angle for RandomRotation
        verbose: If True, print detailed analysis

    Returns:
        dict: Test results including angle synchronization and differences
    """
    # Create test image
    cv_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    cv_image[45:55, 45:55] = [255, 0, 0]  # Red square
    pil_image = Image.fromarray(cv_image)

    # Test angle generation
    torch.manual_seed(42)
    pil_transform = pil_transforms.RandomRotation(max_angle)
    angle_pil = pil_transform.get_params(pil_transform.degrees)

    torch.manual_seed(42)
    cv_transform = transforms.RandomRotation(max_angle)
    angle_cv = cv_transform.get_params(cv_transform.degrees)

    if verbose:
        print(f"\nRandom angle generation (max_angle={max_angle}):")
        print(f"  PIL angle: {angle_pil:.2f}°")
        print(f"  CV angle: {angle_cv:.2f}°")
        print(f"  Angles match: {abs(angle_pil - angle_cv) < 0.001}")

    # Test full transforms
    torch.manual_seed(42)
    random.seed(42)
    pil_rotated = pil_transform(pil_image)

    torch.manual_seed(42)
    random.seed(42)
    cv_rotated = cv_transform(cv_image)

    # Analyze differences
    pil_array = np.array(pil_rotated)
    diff = np.abs(pil_array.astype(float) - cv_rotated.astype(float))

    if verbose:
        print("\nRandom rotation transform:")
        print(f"  Max difference: {diff.max()}")
        print(f"  Mean difference: {diff.mean():.2f}")
        print(
            f"  Pixels > 120: {np.sum(diff > 120)} ({100 * np.sum(diff > 120) / diff.size:.2f}%)"
        )

    return {
        "angle_pil": angle_pil,
        "angle_cv": angle_cv,
        "angles_match": abs(angle_pil - angle_cv) < 0.001,
        "max_diff": diff.max(),
        "mean_diff": diff.mean(),
        "pixels_over_120": np.sum(diff > 120),
    }


def test_interpolation_modes(angle=30, verbose=True):
    """Test different interpolation modes for rotation.

    Args:
        angle: Rotation angle in degrees
        verbose: If True, print detailed analysis

    Returns:
        dict: Results for different interpolation modes
    """
    import cv2

    # Create test image
    cv_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv_image[40:60, 40:60] = [255, 255, 255]  # White square
    pil_image = Image.fromarray(cv_image)

    # Test different interpolation modes
    modes = {
        "NEAREST": cv2.INTER_NEAREST,
        "LINEAR": cv2.INTER_LINEAR,
        "CUBIC": cv2.INTER_CUBIC,
    }

    results = {}

    for mode_name, cv_mode in modes.items():
        # PIL rotation (always uses its default)
        pil_rotated = pil_image.rotate(angle)
        pil_array = np.array(pil_rotated)

        # OpenCV rotation with specific interpolation
        cv_rotated = F.rotate(cv_image, angle, resample=cv_mode)

        # Analyze
        diff = np.abs(pil_array.astype(float) - cv_rotated.astype(float))

        results[mode_name] = {
            "max_diff": diff.max(),
            "mean_diff": diff.mean(),
            "pixels_over_120": np.sum(diff > 120),
        }

        if verbose:
            print(f"\nInterpolation mode: {mode_name}")
            print(f"  Max difference: {diff.max()}")
            print(f"  Mean difference: {diff.mean():.2f}")
            print(f"  Pixels > 120: {np.sum(diff > 120)}")

    return results


def visualize_rotation_difference(angle=10, save_path=None):
    """Create a visual comparison of PIL vs OpenCV rotation.

    Args:
        angle: Rotation angle in degrees
        save_path: If provided, save the visualization to this path

    Returns:
        dict: Images and difference map
    """
    import matplotlib.pyplot as plt

    # Create test image with grid pattern
    size = 200
    cv_image = np.zeros((size, size, 3), dtype=np.uint8)

    # Create grid pattern
    for i in range(0, size, 20):
        cv_image[i : i + 2, :] = [255, 255, 255]
        cv_image[:, i : i + 2] = [255, 255, 255]

    # Add colored squares
    cv_image[50:70, 50:70] = [255, 0, 0]  # Red
    cv_image[130:150, 50:70] = [0, 255, 0]  # Green
    cv_image[50:70, 130:150] = [0, 0, 255]  # Blue
    cv_image[130:150, 130:150] = [255, 255, 0]  # Yellow

    pil_image = Image.fromarray(cv_image)

    # Rotate both ways
    pil_rotated = pil_image.rotate(angle)
    cv_rotated = F.rotate(cv_image, angle)

    # Calculate difference
    pil_array = np.array(pil_rotated)
    diff = np.abs(pil_array.astype(float) - cv_rotated.astype(float))
    diff_map = diff.max(axis=2)  # Max difference across channels

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv_image)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pil_array)
    axes[0, 1].set_title(f"PIL Rotation ({angle}°)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv_rotated)
    axes[0, 2].set_title(f"OpenCV Rotation ({angle}°)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(diff.astype(np.uint8))
    axes[1, 0].set_title("Absolute Difference (RGB)")
    axes[1, 0].axis("off")

    im = axes[1, 1].imshow(diff_map, cmap="hot", vmin=0, vmax=255)
    axes[1, 1].set_title("Difference Heatmap")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1])

    # Histogram of differences
    axes[1, 2].hist(diff.flatten(), bins=50, edgecolor="black")
    axes[1, 2].set_title("Difference Distribution")
    axes[1, 2].set_xlabel("Pixel Difference")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].axvline(x=120, color="r", linestyle="--", label="Threshold (120)")
    axes[1, 2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    return {
        "original": cv_image,
        "pil_rotated": pil_array,
        "cv_rotated": cv_rotated,
        "difference": diff,
        "diff_map": diff_map,
    }


def run_full_analysis():
    """Run a comprehensive analysis of rotation differences."""
    print("=" * 60)
    print("Rotation Transform Analysis: PIL vs OpenCV")
    print("=" * 60)

    # Test 1: Basic rotation with synthetic image
    print("\n1. SYNTHETIC IMAGE TESTS")
    print("-" * 30)
    for angle in [10, 30, 45, 90]:
        analyze_rotation_differences(angle=angle, use_test_image=False, verbose=True)

    # Test 2: Test image from dataset
    print("\n2. BEANS DATASET IMAGE TEST")
    print("-" * 30)
    analyze_rotation_differences(angle=10, use_test_image=True, verbose=True)

    # Test 3: Random rotation synchronization
    print("\n3. RANDOM ROTATION SYNCHRONIZATION")
    print("-" * 30)
    test_random_rotation_sync(max_angle=10, verbose=True)
    test_random_rotation_sync(max_angle=45, verbose=True)

    # Test 4: Interpolation modes
    print("\n4. INTERPOLATION MODE COMPARISON")
    print("-" * 30)
    test_interpolation_modes(angle=30, verbose=True)

    # Test 5: Visual comparison
    print("\n5. GENERATING VISUAL COMPARISON")
    print("-" * 30)
    visualize_rotation_difference(angle=30, save_path="debug/rotation_comparison.png")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_analysis()
