"""Visualization utilities for debugging image transforms.

This module provides functions for creating comparison figures and
visualizing differences between PIL and OpenCV transforms.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. Install with: pip install matplotlib"
        )

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
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. Install with: pip install matplotlib"
        )

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
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. Install with: pip install matplotlib"
        )

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
