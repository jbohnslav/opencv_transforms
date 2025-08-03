import time
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torchvision import transforms as pil_transforms

from opencv_transforms import transforms as cv_transforms


def load_test_samples(num_samples: int = 100) -> List:
    """Load test samples from a public dataset."""
    print("Loading test dataset...")
    # Use a smaller, public dataset for benchmarking
    dataset = load_dataset("beans", split="train", streaming=True)

    samples = []
    print(f"Pre-loading {num_samples} samples...")
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(sample["image"])
        if (i + 1) % 20 == 0:
            print(f"Loaded {i + 1}/{num_samples} samples")

    return samples


def prepare_images(samples: List) -> Tuple[List, List]:
    """Convert samples to both PIL and numpy format."""
    pil_images = samples  # Already PIL images
    cv_images = [np.array(img) for img in samples]  # Convert to numpy
    return pil_images, cv_images


def benchmark_transform(transform_func: Callable, images: List) -> float:
    """Time how long it takes to apply transform to all images."""
    start_time = time.time()

    for img in images:
        _ = transform_func(img)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(images)

    return avg_time


def run_benchmark(
    transform_name: str,
    pil_transform: Callable,
    cv_transform: Callable,
    pil_images: List,
    cv_images: List,
) -> Dict:
    """Run benchmark comparing PIL and OpenCV transforms."""
    print(f"\nBenchmarking {transform_name}...")

    # Benchmark PIL transform
    pil_time = benchmark_transform(pil_transform, pil_images)

    # Benchmark OpenCV transform
    cv_time = benchmark_transform(cv_transform, cv_images)

    speedup = pil_time / cv_time

    print(f"PIL avg time: {pil_time * 1000:.3f} ms")
    print(f"OpenCV avg time: {cv_time * 1000:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")

    return {
        "transform": transform_name,
        "pil_time": pil_time,
        "cv_time": cv_time,
        "speedup": speedup,
    }


def plot_results(results: List[Dict], save_path: Optional[str] = None):
    """Plot benchmark results."""
    transforms = [r["transform"] for r in results]
    pil_times = [r["pil_time"] * 1000 for r in results]  # Convert to ms
    cv_times = [r["cv_time"] * 1000 for r in results]
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


def main():
    # Load test samples
    samples = load_test_samples(num_samples=100)
    pil_images, cv_images = prepare_images(samples)

    # Define transforms to benchmark
    transform_configs = [
        {
            "name": "Resize (256x256)",
            "pil": pil_transforms.Resize((256, 256)),
            "cv": cv_transforms.Resize((256, 256)),
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
            "name": "RandomHorizontalFlip",
            "pil": pil_transforms.RandomHorizontalFlip(p=1.0),
            "cv": cv_transforms.RandomHorizontalFlip(p=1.0),
        },
        {
            "name": "RandomRotation (10°)",
            "pil": pil_transforms.RandomRotation(10),
            "cv": cv_transforms.RandomRotation(10),
        },
        {
            "name": "ColorJitter",
            "pil": pil_transforms.ColorJitter(brightness=0.2, contrast=0.2),
            "cv": cv_transforms.ColorJitter(brightness=0.2, contrast=0.2),
        },
        {
            "name": "RandomResizedCrop (224x224)",
            "pil": pil_transforms.RandomResizedCrop(224),
            "cv": cv_transforms.RandomResizedCrop(224),
        },
        {
            "name": "Grayscale",
            "pil": pil_transforms.Grayscale(num_output_channels=3),
            "cv": cv_transforms.Grayscale(num_output_channels=3),
        },
    ]

    # Run benchmarks
    results = []
    for config in transform_configs:
        result = run_benchmark(
            config["name"], config["pil"], config["cv"], pil_images, cv_images
        )
        results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Transform':<30} {'PIL (ms)':<12} {'OpenCV (ms)':<12} {'Speedup':<10}")
    print("-" * 60)

    for result in results:
        print(
            f"{result['transform']:<30} "
            f"{result['pil_time'] * 1000:<12.3f} "
            f"{result['cv_time'] * 1000:<12.3f} "
            f"{result['speedup']:<10.2f}x"
        )

    avg_speedup = np.mean([r["speedup"] for r in results])
    print("-" * 60)
    print(f"{'Average Speedup:':<30} {'':<12} {'':<12} {avg_speedup:<10.2f}x")

    # Plot results
    plot_results(results, save_path="imagenet_benchmark_results.png")


if __name__ == "__main__":
    main()
