"""Benchmark script for comparing PIL and OpenCV transform performance."""

import random
import time
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms as pil_transforms

from opencv_transforms import transforms as cv_transforms


def load_cityscapes_dataset(num_samples: int = 100) -> List[Image.Image]:
    """Load Cityscapes dataset images for benchmarking."""
    print("Loading Cityscapes dataset...")
    try:
        # Try to load Cityscapes dataset
        dataset = load_dataset("Chris1/cityscapes", split="validation", streaming=True)
        print("Using Cityscapes dataset for benchmarking")

        samples = []
        print(f"Loading {num_samples} samples...")
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            samples.append(sample["image"])
            if (i + 1) % 20 == 0:
                print(f"Loaded {i + 1}/{num_samples} samples")

        return samples

    except Exception as e:
        print(f"Failed to load Cityscapes dataset: {e}")
        print("Creating synthetic large images as fallback...")
        # Create synthetic images as fallback (large size for Cityscapes replacement)
        samples = []
        for _ in range(num_samples):
            # Create random RGB image similar to Cityscapes size
            np_img = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
            pil_img = Image.fromarray(np_img)
            samples.append(pil_img)

        return samples


def load_imagenet_validation(num_samples: int = 100) -> List[Image.Image]:
    """Load ImageNet validation set images for benchmarking."""
    print("Loading ImageNet validation dataset...")
    try:
        # Try to load ImageNet validation set
        dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
        print("Using ImageNet validation set for benchmarking")

        samples = []
        print(f"Loading {num_samples} samples...")
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            # Keep original resolution - don't resize to 224x224
            samples.append(sample["image"])
            if (i + 1) % 20 == 0:
                print(f"Loaded {i + 1}/{num_samples} samples")

        return samples

    except Exception as e:
        print(f"Failed to load ImageNet dataset: {e}")
        print("Creating synthetic ImageNet-sized images as fallback...")
        # Create synthetic images as fallback with varied ImageNet-like sizes
        samples = []
        imagenet_sizes = [(480, 640), (375, 500), (224, 224), (256, 256), (299, 299)]
        for i in range(num_samples):
            # Use different sizes to simulate real ImageNet variety
            h, w = imagenet_sizes[i % len(imagenet_sizes)]
            np_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            pil_img = Image.fromarray(np_img)
            samples.append(pil_img)

        return samples


def prepare_multi_size_images(
    cityscapes_samples: List[Image.Image], imagenet_samples: List[Image.Image]
) -> Dict[Tuple[int, int], Tuple[List[Image.Image], List[np.ndarray]]]:
    """Prepare images at multiple sizes for benchmarking.

    Args:
        cityscapes_samples: Large resolution images from Cityscapes
        imagenet_samples: Various resolution images from ImageNet

    Returns:
        Dictionary mapping (width, height) -> (pil_images, cv_images)
    """
    sizes_and_images = {}

    # Define target sizes: Cityscapes for large, ImageNet for small
    target_sizes = [
        (2048, 1024),  # Large - from Cityscapes
        (1024, 512),  # Large - from Cityscapes
        (512, 256),  # Large - from Cityscapes
        (256, 256),  # Small - from ImageNet
        (224, 224),  # Small - from ImageNet
    ]

    for width, height in target_sizes:
        print(f"Preparing images at size {width}x{height}...")

        # Choose source dataset based on target size
        source_samples = cityscapes_samples if width >= 512 else imagenet_samples

        # Resize all source images to target size
        pil_images = []
        for img in source_samples:
            resized_img = img.resize((width, height), Image.LANCZOS)
            pil_images.append(resized_img)

        # Convert to OpenCV format
        cv_images = [np.array(img) for img in pil_images]

        sizes_and_images[(width, height)] = (pil_images, cv_images)

    return sizes_and_images


def prepare_images(
    samples: List[Image.Image],
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """Convert samples to both PIL and numpy format. (Legacy function)"""
    pil_images = samples
    cv_images = [np.array(img) for img in samples]
    return pil_images, cv_images


def benchmark_transform(
    transform_func: Callable, images: List
) -> Tuple[float, float, float, float]:
    """Time how long it takes to apply transform to all images.

    Returns:
        Tuple of (avg_time, std_time, min_time, max_time)
    """
    times = []

    for img in images:
        start_time = time.time()
        try:
            result = transform_func(img)
            # Handle transforms that return tuples (FiveCrop, TenCrop)
            if isinstance(result, tuple):
                pass  # Just measuring time, don't need to process result
        except Exception as e:
            print(f"Warning: Transform failed on an image: {e}")
            continue
        end_time = time.time()
        times.append(end_time - start_time)

    if not times:
        return 0.0, 0.0, 0.0, 0.0

    avg_time = np.mean(times)
    std_time = np.std(times) if len(times) > 1 else 0.0
    min_time = np.min(times)
    max_time = np.max(times)

    return avg_time, std_time, min_time, max_time


def run_benchmark(
    transform_name: str,
    pil_transform: Callable,
    cv_transform: Callable,
    pil_images: List,
    cv_images: List,
    image_size: Tuple[int, int],
) -> Dict:
    """Run benchmark comparing PIL and OpenCV transforms at a specific image size."""
    width, height = image_size
    print(f"\nBenchmarking {transform_name} at {width}x{height}...")

    # Benchmark PIL transform
    pil_avg, pil_std, pil_min, pil_max = benchmark_transform(pil_transform, pil_images)

    # Benchmark OpenCV transform
    cv_avg, cv_std, cv_min, cv_max = benchmark_transform(cv_transform, cv_images)

    # Calculate speedup (handle division by zero)
    speedup = pil_avg / cv_avg if cv_avg > 0 else 0.0

    print(f"PIL avg time: {pil_avg * 1000:.3f} ms (std: {pil_std * 1000:.3f})")
    print(f"OpenCV avg time: {cv_avg * 1000:.3f} ms (std: {cv_std * 1000:.3f})")
    print(f"Speedup: {speedup:.2f}x")

    return {
        "transform": transform_name,
        "image_size": image_size,
        "width": width,
        "height": height,
        "pixel_count": width * height,
        "pil_avg_time": pil_avg,
        "pil_std_time": pil_std,
        "pil_min_time": pil_min,
        "pil_max_time": pil_max,
        "cv_avg_time": cv_avg,
        "cv_std_time": cv_std,
        "cv_min_time": cv_min,
        "cv_max_time": cv_max,
        "speedup": speedup,
    }


def plot_results(results: List[Dict], save_path: Optional[str] = None):
    """Plot benchmark results."""
    transforms = [r["transform"] for r in results]
    pil_times = [r["pil_avg_time"] * 1000 for r in results]  # Convert to ms
    cv_times = [r["cv_avg_time"] * 1000 for r in results]
    speedups = [r["speedup"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot timing comparison
    x = np.arange(len(transforms))
    width = 0.35

    ax1.bar(x - width / 2, pil_times, width, label="PIL", color="blue", alpha=0.7)
    ax1.bar(x + width / 2, cv_times, width, label="OpenCV", color="green", alpha=0.7)
    ax1.set_xlabel("Transform")
    ax1.set_ylabel("Average Time (ms)")
    ax1.set_title("Transform Execution Time Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(transforms, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot speedup
    ax2.bar(x, speedups, color="orange", alpha=0.7)
    ax2.set_xlabel("Transform")
    ax2.set_ylabel("Speedup Factor")
    ax2.set_title("OpenCV Speedup over PIL")
    ax2.set_xticks(x)
    ax2.set_xticklabels(transforms, rotation=45, ha="right")
    ax2.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="No speedup")
    ax2.grid(True, alpha=0.3)

    # Add speedup values on bars
    for i, v in enumerate(speedups):
        ax2.text(i, v + 0.1, f"{v:.2f}x", ha="center", va="bottom")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def get_transform_configs():
    """Get configuration for transforms to benchmark."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    return [
        # Basic transforms
        {
            "name": "Resize (224x224)",
            "pil": pil_transforms.Resize((224, 224)),
            "cv": cv_transforms.Resize((224, 224)),
        },
        {
            "name": "CenterCrop (224x224)",
            "pil": pil_transforms.CenterCrop(224),
            "cv": cv_transforms.CenterCrop(224),
        },
        {
            "name": "RandomCrop (224x224)",
            "pil": pil_transforms.RandomCrop(224),
            "cv": cv_transforms.RandomCrop(224),
        },
        {
            "name": "RandomResizedCrop (224x224)",
            "pil": pil_transforms.RandomResizedCrop(224),
            "cv": cv_transforms.RandomResizedCrop(224),
        },
        # Flipping transforms
        {
            "name": "RandomHorizontalFlip (p=1.0)",
            "pil": pil_transforms.RandomHorizontalFlip(p=1.0),
            "cv": cv_transforms.RandomHorizontalFlip(p=1.0),
        },
        {
            "name": "RandomVerticalFlip (p=1.0)",
            "pil": pil_transforms.RandomVerticalFlip(p=1.0),
            "cv": cv_transforms.RandomVerticalFlip(p=1.0),
        },
        # Padding transform
        {
            "name": "Pad (padding=10)",
            "pil": pil_transforms.Pad(10, fill=0),
            "cv": cv_transforms.Pad(10, fill=0),
        },
        # Color transforms
        {
            "name": "ColorJitter (brightness=0.2)",
            "pil": pil_transforms.ColorJitter(brightness=0.2),
            "cv": cv_transforms.ColorJitter(brightness=0.2),
        },
        {
            "name": "ColorJitter (all params)",
            "pil": pil_transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            "cv": cv_transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
        },
        {
            "name": "Grayscale (3 channels)",
            "pil": pil_transforms.Grayscale(num_output_channels=3),
            "cv": cv_transforms.Grayscale(num_output_channels=3),
        },
        {
            "name": "RandomGrayscale (p=1.0)",
            "pil": pil_transforms.RandomGrayscale(p=1.0),
            "cv": cv_transforms.RandomGrayscale(p=1.0),
        },
        # Geometric transforms
        {
            "name": "RandomRotation (15°)",
            "pil": pil_transforms.RandomRotation(15),
            "cv": cv_transforms.RandomRotation(15),
        },
        {
            "name": "RandomAffine (degrees=15)",
            "pil": pil_transforms.RandomAffine(degrees=15),
            "cv": cv_transforms.RandomAffine(degrees=15),
        },
        # Tensor operations
        {
            "name": "ToTensor",
            "pil": pil_transforms.ToTensor(),
            "cv": cv_transforms.ToTensor(),
        },
        # Multi-crop transforms
        {
            "name": "FiveCrop (224x224)",
            "pil": pil_transforms.FiveCrop(224),
            "cv": cv_transforms.FiveCrop(224),
        },
        {
            "name": "TenCrop (224x224)",
            "pil": pil_transforms.TenCrop(224),
            "cv": cv_transforms.TenCrop(224),
        },
        # Composite transforms
        {
            "name": "Standard ImageNet Pipeline",
            "pil": pil_transforms.Compose(
                [
                    pil_transforms.Resize(256),
                    pil_transforms.CenterCrop(224),
                    pil_transforms.ToTensor(),
                    pil_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "cv": cv_transforms.Compose(
                [
                    cv_transforms.Resize(256),
                    cv_transforms.CenterCrop(224),
                    cv_transforms.ToTensor(),
                    cv_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        },
    ]


def main():
    """Main benchmarking function."""
    # Load datasets
    print("Loading datasets...")
    cityscapes_samples = load_cityscapes_dataset(num_samples=100)
    imagenet_samples = load_imagenet_validation(num_samples=100)

    # Prepare multi-size images
    print("\nPreparing images at multiple sizes...")
    size_dict = prepare_multi_size_images(cityscapes_samples, imagenet_samples)
    sizes = sorted(
        size_dict.keys(), key=lambda x: x[0] * x[1], reverse=True
    )  # Sort by pixel count
    print(f"Using {len(sizes)} image sizes: {[f'{w}x{h}' for w, h in sizes]}")

    # Get transform configurations
    transform_configs = get_transform_configs()
    print(f"Benchmarking {len(transform_configs)} transforms across {len(sizes)} sizes")
    print(f"Total benchmark runs: {len(transform_configs) * len(sizes)}")

    # Run benchmarks for all transforms and sizes
    results = []
    total_runs = len(transform_configs) * len(sizes)
    current_run = 0

    for size in sizes:
        width, height = size
        pil_images, cv_images = size_dict[size]
        print(f"\n{'=' * 60}")
        print(f"BENCHMARKING AT SIZE: {width}x{height} ({width * height:,} pixels)")
        print(f"{'=' * 60}")

        for config in transform_configs:
            current_run += 1
            print(f"[{current_run}/{total_runs}] ", end="")

            result = run_benchmark(
                config["name"], config["pil"], config["cv"], pil_images, cv_images, size
            )
            results.append(result)

    # Print detailed summary by size
    print("\n" + "=" * 90)
    print("DETAILED BENCHMARK SUMMARY")
    print("=" * 90)

    for size in sizes:
        width, height = size
        print(f"\nImage Size: {width}x{height} ({width * height:,} pixels)")
        print("-" * 70)
        print(f"{'Transform':<35} {'PIL (ms)':<12} {'OpenCV (ms)':<12} {'Speedup':<10}")
        print("-" * 70)

        size_results = [r for r in results if r["image_size"] == size]
        for result in size_results:
            print(
                f"{result['transform']:<35} "
                f"{result['pil_avg_time'] * 1000:<12.3f} "
                f"{result['cv_avg_time'] * 1000:<12.3f} "
                f"{result['speedup']:<10.2f}x"
            )

        # Size-specific average speedup
        valid_speedups = [r["speedup"] for r in size_results if r["speedup"] > 0]
        if valid_speedups:
            avg_speedup = np.mean(valid_speedups)
            print("-" * 70)
            print(
                f"{'Size Average Speedup:':<35} {'':<12} {'':<12} {avg_speedup:<10.2f}x"
            )

    # Overall summary
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)
    valid_speedups = [r["speedup"] for r in results if r["speedup"] > 0]
    if valid_speedups:
        overall_avg_speedup = np.mean(valid_speedups)
        print(f"Total benchmark runs: {len(results)}")
        print(f"Overall average speedup: {overall_avg_speedup:.2f}x")

    # Generate and save results plot (will be updated for multi-size)
    plot_results(results, save_path="multi_size_benchmark_results.png")


if __name__ == "__main__":
    main()
