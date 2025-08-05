"""Visualization utilities for debugging image transforms.

This module provides functions for creating comparison figures and
visualizing differences between PIL and OpenCV transforms.
"""

import matplotlib.pyplot as plt
import numpy as np


def create_comparison_figure(
    original, pil_result, cv_result, transform_name="Transform", save_path=None
):
    """Create a side-by-side comparison figure of transform results.

    Args:
        original: Original image as numpy array
        pil_result: PIL transform result as numpy array
        cv_result: OpenCV transform result as numpy array
        transform_name: Name of the transform for the title
        save_path: Optional path to save the figure

    Raises:
        ImportError: If matplotlib is not available
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original
    axes[0, 0].imshow(original, cmap="gray" if original.ndim == 2 else None)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # PIL result
    axes[0, 1].imshow(pil_result, cmap="gray" if pil_result.ndim == 2 else None)
    axes[0, 1].set_title("PIL Result")
    axes[0, 1].axis("off")

    # OpenCV result
    axes[1, 0].imshow(cv_result, cmap="gray" if cv_result.ndim == 2 else None)
    axes[1, 0].set_title("OpenCV Result")
    axes[1, 0].axis("off")

    # Difference map
    if pil_result.shape == cv_result.shape:
        diff = np.abs(pil_result.astype(float) - cv_result.astype(float))
        im = axes[1, 1].imshow(diff, cmap="hot")
        axes[1, 1].set_title("Absolute Difference")
        axes[1, 1].axis("off")
        plt.colorbar(im, ax=axes[1, 1])
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Different shapes",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].axis("off")

    plt.suptitle(f"{transform_name} Comparison")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_pixel_distribution(pil_result, cv_result, bins=50):
    """Plot histogram comparison of pixel value distributions.

    Args:
        pil_result: PIL result as numpy array
        cv_result: OpenCV result as numpy array
        bins: Number of histogram bins

    Raises:
        ImportError: If matplotlib is not available
    """

    plt.figure(figsize=(10, 6))

    # Flatten arrays
    pil_flat = pil_result.flatten()
    cv_flat = cv_result.flatten()

    # Plot histograms
    plt.hist(pil_flat, bins=bins, alpha=0.5, label="PIL", density=True)
    plt.hist(cv_flat, bins=bins, alpha=0.5, label="OpenCV", density=True)

    plt.xlabel("Pixel Value")
    plt.ylabel("Density")
    plt.title("Pixel Value Distribution Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_difference_heatmap(pil_result, cv_result, threshold=1):
    """Create a heatmap showing where images differ.

    Args:
        pil_result: PIL result as numpy array
        cv_result: OpenCV result as numpy array
        threshold: Minimum difference to highlight (default: 1)

    Raises:
        ImportError: If matplotlib is not available
    """

    if pil_result.shape != cv_result.shape:
        print("Error: Images have different shapes")
        return

    diff = np.abs(pil_result.astype(float) - cv_result.astype(float))
    diff_binary = diff >= threshold

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Difference magnitude
    im1 = ax1.imshow(diff, cmap="hot")
    ax1.set_title("Difference Magnitude")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1)

    # Binary difference map
    ax2.imshow(diff_binary, cmap="binary")
    ax2.set_title(f"Pixels with difference >= {threshold}")
    ax2.axis("off")

    # Add statistics
    num_diff = np.sum(diff_binary)
    total = diff.size
    percent = (num_diff / total) * 100
    fig.suptitle(
        f"Difference Analysis: {num_diff}/{total} pixels differ ({percent:.2f}%)"
    )

    plt.tight_layout()
    plt.show()


def create_rotation_comparison_figure(
    original, pil_result, cv_result, angle, save_path=None
):
    """Create a comparison figure specifically for rotation transforms.

    Args:
        original: Original image as numpy array
        pil_result: PIL rotation result as numpy array
        cv_result: OpenCV rotation result as numpy array
        angle: Rotation angle in degrees
        save_path: Optional path to save the figure

    Raises:
        ImportError: If matplotlib is not available
    """

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Calculate difference and stats
    diff = np.abs(pil_result.astype(float) - cv_result.astype(float))
    max_diff = diff.max()
    mean_diff = diff.mean()
    num_diff = np.count_nonzero(diff > 0.1)
    total_pixels = diff.size

    # Top row: Images
    axes[0, 0].imshow(original, cmap="gray" if original.ndim == 2 else None)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pil_result, cmap="gray" if pil_result.ndim == 2 else None)
    axes[0, 1].set_title(f"PIL {angle}°")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv_result, cmap="gray" if cv_result.ndim == 2 else None)
    axes[0, 2].set_title(f"OpenCV {angle}°")
    axes[0, 2].axis("off")

    # Bottom row: Analysis
    # Difference heatmap
    im1 = axes[1, 0].imshow(
        diff[:, :, 0] if diff.ndim == 3 else diff, cmap="hot", vmin=0, vmax=255
    )
    axes[1, 0].set_title("Difference Heatmap")
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0])

    # Enhanced difference (5x)
    diff_enhanced = np.clip(diff * 5, 0, 255)
    axes[1, 1].imshow(
        diff_enhanced[:, :, 0] if diff_enhanced.ndim == 3 else diff_enhanced, cmap="hot"
    )
    axes[1, 1].set_title("Difference x5")
    axes[1, 1].axis("off")

    # Statistics
    axes[1, 2].text(
        0.1,
        0.8,
        f"Rotation: {angle}°\n\n"
        f"Max difference: {max_diff:.1f}\n"
        f"Mean difference: {mean_diff:.3f}\n"
        f"Differing pixels: {num_diff:,}\n"
        f"Percentage: {num_diff / total_pixels * 100:.2f}%",
        transform=axes[1, 2].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
    )
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis("off")

    plt.suptitle(f"Rotation Comparison: {angle}°", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_coordinate_system_comparison(
    old_result, new_result, pil_result, angle, save_path=None
):
    """Create a before/after comparison showing coordinate system fix impact.

    Args:
        old_result: OpenCV result with old coordinate system
        new_result: OpenCV result with new coordinate system
        pil_result: PIL result (ground truth)
        angle: Rotation angle in degrees
        save_path: Optional path to save the figure

    Raises:
        ImportError: If matplotlib is not available
    """

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Calculate differences
    diff_old = np.abs(pil_result.astype(float) - old_result.astype(float))
    diff_new = np.abs(pil_result.astype(float) - new_result.astype(float))

    # Top row: Before fix
    axes[0, 0].imshow(pil_result, cmap="gray" if pil_result.ndim == 2 else None)
    axes[0, 0].set_title("PIL (Target)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(old_result, cmap="gray" if old_result.ndim == 2 else None)
    axes[0, 1].set_title("OpenCV (Old Coords)")
    axes[0, 1].axis("off")

    diff_old_enhanced = np.clip(diff_old * 5, 0, 255)
    axes[0, 2].imshow(
        diff_old_enhanced[:, :, 0]
        if diff_old_enhanced.ndim == 3
        else diff_old_enhanced,
        cmap="hot",
    )
    axes[0, 2].set_title(f"Old Diff x5\nMax: {diff_old.max():.1f}")
    axes[0, 2].axis("off")

    # Stats for old system
    num_diff_old = np.count_nonzero(diff_old > 0.1)
    axes[0, 3].text(
        0.1,
        0.7,
        f"BEFORE FIX:\nMax diff: {diff_old.max():.1f}\n"
        f"Mean diff: {diff_old.mean():.3f}\n"
        f"Differing pixels: {num_diff_old:,}",
        transform=axes[0, 3].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightcoral", "alpha": 0.7},
    )
    axes[0, 3].set_xlim(0, 1)
    axes[0, 3].set_ylim(0, 1)
    axes[0, 3].axis("off")

    # Bottom row: After fix
    axes[1, 0].imshow(pil_result, cmap="gray" if pil_result.ndim == 2 else None)
    axes[1, 0].set_title("PIL (Target)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(new_result, cmap="gray" if new_result.ndim == 2 else None)
    axes[1, 1].set_title("OpenCV (Fixed Coords)")
    axes[1, 1].axis("off")

    diff_new_enhanced = np.clip(diff_new * 5, 0, 255)
    axes[1, 2].imshow(
        diff_new_enhanced[:, :, 0]
        if diff_new_enhanced.ndim == 3
        else diff_new_enhanced,
        cmap="hot",
    )
    axes[1, 2].set_title(f"New Diff x5\nMax: {diff_new.max():.1f}")
    axes[1, 2].axis("off")

    # Stats for new system
    num_diff_new = np.count_nonzero(diff_new > 0.1)
    improvement = num_diff_old - num_diff_new
    axes[1, 3].text(
        0.1,
        0.7,
        f"AFTER FIX:\nMax diff: {diff_new.max():.1f}\n"
        f"Mean diff: {diff_new.mean():.3f}\n"
        f"Differing pixels: {num_diff_new:,}\n\n"
        f"IMPROVEMENT:\n{improvement:,} fewer pixels",
        transform=axes[1, 3].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.7},
    )
    axes[1, 3].set_xlim(0, 1)
    axes[1, 3].set_ylim(0, 1)
    axes[1, 3].axis("off")

    plt.suptitle(f"Coordinate System Fix Comparison: {angle}°", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_rotation_test_summary_figure(test_results, save_path=None):
    """Create a summary figure showing rotation test results across angles.

    Args:
        test_results: List of test results from test_rotation_angles()
        save_path: Optional path to save the figure

    Raises:
        ImportError: If matplotlib is not available
    """

    angles = [r["angle"] for r in test_results]
    max_diffs = [r["max_diff"] for r in test_results]
    mean_diffs = [r["mean_diff"] for r in test_results]
    passes = [r["passes"] for r in test_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Max differences plot
    colors = ["green" if p else "red" for p in passes]
    bars1 = ax1.bar(angles, max_diffs, color=colors, alpha=0.7)
    ax1.axhline(y=220.0, color="red", linestyle="--", label="Tolerance (220.0)")
    ax1.set_xlabel("Rotation Angle (degrees)")
    ax1.set_ylabel("Maximum Pixel Difference")
    ax1.set_title("Maximum Differences by Angle")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, diff in zip(bars1, max_diffs):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            f"{diff:.1f}",
            ha="center",
            va="bottom",
        )

    # Mean differences plot
    ax2.plot(angles, mean_diffs, "bo-", linewidth=2, markersize=8)
    ax2.set_xlabel("Rotation Angle (degrees)")
    ax2.set_ylabel("Mean Pixel Difference")
    ax2.set_title("Mean Differences by Angle")
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for angle, diff in zip(angles, mean_diffs):
        ax2.text(
            angle,
            diff + max(mean_diffs) * 0.05,
            f"{diff:.3f}",
            ha="center",
            va="bottom",
        )

    # Summary text
    passed = sum(passes)
    total = len(passes)
    fig.suptitle(f"Rotation Test Summary: {passed}/{total} angles passed", fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def create_random_resized_crop_comparison(
    pil_result, cv_result, parameters, save_path=None
):
    """Create a comparison visualization for RandomResizedCrop outputs.

    Args:
        pil_result: PIL RandomResizedCrop result (numpy array)
        cv_result: OpenCV RandomResizedCrop result (numpy array)
        parameters: Dict with crop parameters and metadata
        save_path: Optional path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """

    # Ensure inputs are numpy arrays
    pil_array = (
        np.array(pil_result) if not isinstance(pil_result, np.ndarray) else pil_result
    )
    cv_array = (
        np.array(cv_result) if not isinstance(cv_result, np.ndarray) else cv_result
    )

    # Calculate difference
    diff = np.abs(pil_array.astype(np.float32) - cv_array.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"RandomResizedCrop Comparison (Max Diff: {max_diff:.1f})", fontsize=16
    )

    # Top row: PIL, OpenCV, Difference
    axes[0, 0].imshow(pil_array)
    axes[0, 0].set_title("PIL RandomResizedCrop")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv_array)
    axes[0, 1].set_title("OpenCV RandomResizedCrop")
    axes[0, 1].axis("off")

    # Difference heatmap
    diff_display = np.mean(diff, axis=2) if len(diff.shape) == 3 else diff
    im_diff = axes[0, 2].imshow(diff_display, cmap="hot", vmin=0, vmax=max_diff)
    axes[0, 2].set_title(f"Difference (Max: {max_diff:.1f})")
    axes[0, 2].axis("off")
    plt.colorbar(im_diff, ax=axes[0, 2])

    # Bottom row: Statistics and crop region info
    axes[1, 0].axis("off")
    stats_text = [
        f"Max Difference: {max_diff:.2f}",
        f"Mean Difference: {mean_diff:.2f}",
        f"PIL Shape: {pil_array.shape}",
        f"OpenCV Shape: {cv_array.shape}",
        "",
        "Crop Parameters:",
        f"  i={parameters.get('i', 'N/A')}, j={parameters.get('j', 'N/A')}",
        f"  h={parameters.get('h', 'N/A')}, w={parameters.get('w', 'N/A')}",
        f"  Scale: {parameters.get('scale', 'N/A')}",
        f"  Size: {parameters.get('size', 'N/A')}",
    ]
    axes[1, 0].text(
        0.1,
        0.9,
        "\n".join(stats_text),
        transform=axes[1, 0].transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
    )

    # Difference histogram
    diff_flat = diff.flatten()
    axes[1, 1].hist(diff_flat, bins=50, alpha=0.7, color="red")
    axes[1, 1].set_title("Difference Distribution")
    axes[1, 1].set_xlabel("Pixel Difference")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].axvline(
        mean_diff, color="blue", linestyle="--", label=f"Mean: {mean_diff:.2f}"
    )
    axes[1, 1].legend()

    # Sample comparison (center region)
    center_y, center_x = pil_array.shape[0] // 2, pil_array.shape[1] // 2
    sample_size = 20
    y_start, y_end = (
        max(0, center_y - sample_size // 2),
        min(pil_array.shape[0], center_y + sample_size // 2),
    )
    x_start, x_end = (
        max(0, center_x - sample_size // 2),
        min(pil_array.shape[1], center_x + sample_size // 2),
    )

    pil_sample = pil_array[y_start:y_end, x_start:x_end]
    diff_sample = diff[y_start:y_end, x_start:x_end]

    # Show sample region with difference overlay
    axes[1, 2].imshow(pil_sample)
    if len(diff_sample.shape) == 3:
        diff_sample_gray = np.mean(diff_sample, axis=2)
    else:
        diff_sample_gray = diff_sample
    axes[1, 2].contour(
        diff_sample_gray, levels=[10, 50, 100], colors="yellow", linewidths=1
    )
    axes[1, 2].set_title("Center Sample (Diff Contours)")
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_random_resized_crop_parameter_comparison(parameter_results, save_path=None):
    """Create visualization comparing RandomResizedCrop parameters across seeds.

    Args:
        parameter_results: List of parameter results from debug_random_resized_crop_parameters
        save_path: Optional path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("RandomResizedCrop Parameter Analysis", fontsize=16)

    # Extract parameter data
    seeds = [r["seed"] for r in parameter_results]
    i_values = [r["i"] for r in parameter_results]
    j_values = [r["j"] for r in parameter_results]
    h_values = [r["h"] for r in parameter_results]
    w_values = [r["w"] for r in parameter_results]

    # Plot parameter distributions
    axes[0, 0].bar(range(len(seeds)), i_values, alpha=0.7, color="red")
    axes[0, 0].set_title("Crop Top Position (i)")
    axes[0, 0].set_xlabel("Seed Index")
    axes[0, 0].set_ylabel("i value")
    axes[0, 0].set_xticks(range(len(seeds)))
    axes[0, 0].set_xticklabels([str(s) for s in seeds], rotation=45)

    axes[0, 1].bar(range(len(seeds)), j_values, alpha=0.7, color="green")
    axes[0, 1].set_title("Crop Left Position (j)")
    axes[0, 1].set_xlabel("Seed Index")
    axes[0, 1].set_ylabel("j value")
    axes[0, 1].set_xticks(range(len(seeds)))
    axes[0, 1].set_xticklabels([str(s) for s in seeds], rotation=45)

    axes[1, 0].bar(range(len(seeds)), h_values, alpha=0.7, color="blue")
    axes[1, 0].set_title("Crop Height (h)")
    axes[1, 0].set_xlabel("Seed Index")
    axes[1, 0].set_ylabel("h value")
    axes[1, 0].set_xticks(range(len(seeds)))
    axes[1, 0].set_xticklabels([str(s) for s in seeds], rotation=45)

    axes[1, 1].bar(range(len(seeds)), w_values, alpha=0.7, color="orange")
    axes[1, 1].set_title("Crop Width (w)")
    axes[1, 1].set_xlabel("Seed Index")
    axes[1, 1].set_ylabel("w value")
    axes[1, 1].set_xticks(range(len(seeds)))
    axes[1, 1].set_xticklabels([str(s) for s in seeds], rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
