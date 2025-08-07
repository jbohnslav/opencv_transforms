import random
import time
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageEnhance
import torch
from datasets import load_dataset
from PIL import Image
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
) -> Dict:
    """Run benchmark comparing PIL and OpenCV transforms."""
    print(f"\nBenchmarking {transform_name}...")

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


def get_all_transform_configs():
    """Get configuration for all transforms to benchmark."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    transform_configs = []

    # Task 4: Basic transforms
    # Compose
    resize_256 = {
        "pil": pil_transforms.Resize((256, 256)),
        "cv": cv_transforms.Resize((256, 256)),
    }
    center_crop_224 = {
        "pil": pil_transforms.CenterCrop(224),
        "cv": cv_transforms.CenterCrop(224),
    }
    transform_configs.append(
        {
            "name": "Compose (Resize+CenterCrop)",
            "pil": pil_transforms.Compose([resize_256["pil"], center_crop_224["pil"]]),
            "cv": cv_transforms.Compose([resize_256["cv"], center_crop_224["cv"]]),
        }
    )

    # ToTensor
    transform_configs.append(
        {
            "name": "ToTensor",
            "pil": pil_transforms.ToTensor(),
            "cv": cv_transforms.ToTensor(),
        }
    )

    # Normalize (needs tensor input, so compose with ToTensor)
    transform_configs.append(
        {
            "name": "Normalize (with ToTensor)",
            "pil": pil_transforms.Compose(
                [
                    pil_transforms.ToTensor(),
                    pil_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "cv": cv_transforms.Compose(
                [
                    cv_transforms.ToTensor(),
                    cv_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        }
    )

    # Task 5: Resizing transforms
    transform_configs.append(
        {
            "name": "Resize (256x256)",
            "pil": pil_transforms.Resize((256, 256)),
            "cv": cv_transforms.Resize((256, 256)),
        }
    )

    # Scale is deprecated and removed in newer torchvision
    # Using Resize instead for testing the OpenCV Scale transform
    transform_configs.append(
        {
            "name": "Scale (deprecated, 256)",
            "pil": pil_transforms.Resize(256),  # Use Resize as Scale is removed
            "cv": cv_transforms.Scale(256),  # Test our Scale implementation
        }
    )

    transform_configs.append(
        {
            "name": "RandomResizedCrop (224x224)",
            "pil": pil_transforms.RandomResizedCrop(224),
            "cv": cv_transforms.RandomResizedCrop(224),
        }
    )

    # RandomSizedCrop is deprecated and removed in newer torchvision
    # Using RandomResizedCrop instead for testing the OpenCV RandomSizedCrop transform
    transform_configs.append(
        {
            "name": "RandomSizedCrop (deprecated, 224)",
            "pil": pil_transforms.RandomResizedCrop(
                224
            ),  # Use RandomResizedCrop as RandomSizedCrop is removed
            "cv": cv_transforms.RandomSizedCrop(
                224
            ),  # Test our RandomSizedCrop implementation
        }
    )

    # Task 6: Cropping transforms
    transform_configs.append(
        {
            "name": "CenterCrop (224x224)",
            "pil": pil_transforms.CenterCrop(224),
            "cv": cv_transforms.CenterCrop(224),
        }
    )

    transform_configs.append(
        {
            "name": "RandomCrop (224x224)",
            "pil": pil_transforms.RandomCrop(224),
            "cv": cv_transforms.RandomCrop(224),
        }
    )

    transform_configs.append(
        {
            "name": "FiveCrop (224x224)",
            "pil": pil_transforms.FiveCrop(224),
            "cv": cv_transforms.FiveCrop(224),
        }
    )

    transform_configs.append(
        {
            "name": "TenCrop (224x224)",
            "pil": pil_transforms.TenCrop(224),
            "cv": cv_transforms.TenCrop(224),
        }
    )

    # Task 7: Flipping transforms
    transform_configs.append(
        {
            "name": "RandomHorizontalFlip (p=1.0)",
            "pil": pil_transforms.RandomHorizontalFlip(p=1.0),
            "cv": cv_transforms.RandomHorizontalFlip(p=1.0),
        }
    )

    transform_configs.append(
        {
            "name": "RandomVerticalFlip (p=1.0)",
            "pil": pil_transforms.RandomVerticalFlip(p=1.0),
            "cv": cv_transforms.RandomVerticalFlip(p=1.0),
        }
    )

    # Task 8: Padding transform
    transform_configs.append(
        {
            "name": "Pad (padding=10)",
            "pil": pil_transforms.Pad(10, fill=0),
            "cv": cv_transforms.Pad(10, fill=0),
        }
    )

    transform_configs.append(
        {
            "name": "Pad (padding=(10,20,30,40))",
            "pil": pil_transforms.Pad((10, 20, 30, 40), fill=128),
            "cv": cv_transforms.Pad((10, 20, 30, 40), fill=128),
        }
    )

    # Task 9: Color transforms
    transform_configs.append(
        {
            "name": "ColorJitter (brightness=0.2)",
            "pil": pil_transforms.ColorJitter(brightness=0.2),
            "cv": cv_transforms.ColorJitter(brightness=0.2),
        }
    )

    transform_configs.append(
        {
            "name": "ColorJitter (all params)",
            "pil": pil_transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            "cv": cv_transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
        }
    )

    transform_configs.append(
        {
            "name": "Grayscale (3 channels)",
            "pil": pil_transforms.Grayscale(num_output_channels=3),
            "cv": cv_transforms.Grayscale(num_output_channels=3),
        }
    )

    transform_configs.append(
        {
            "name": "Grayscale (1 channel)",
            "pil": pil_transforms.Grayscale(num_output_channels=1),
            "cv": cv_transforms.Grayscale(num_output_channels=1),
        }
    )

    transform_configs.append(
        {
            "name": "RandomGrayscale (p=1.0)",
            "pil": pil_transforms.RandomGrayscale(p=1.0),
            "cv": cv_transforms.RandomGrayscale(p=1.0),
        }
    )

    # Task 10: Geometric transforms
    transform_configs.append(
        {
            "name": "RandomRotation (10°)",
            "pil": pil_transforms.RandomRotation(10),
            "cv": cv_transforms.RandomRotation(10),
        }
    )

    transform_configs.append(
        {
            "name": "RandomRotation ((-30, 30))",
            "pil": pil_transforms.RandomRotation((-30, 30)),
            "cv": cv_transforms.RandomRotation((-30, 30)),
        }
    )

    transform_configs.append(
        {
            "name": "RandomAffine (degrees=10)",
            "pil": pil_transforms.RandomAffine(degrees=10),
            "cv": cv_transforms.RandomAffine(degrees=10),
        }
    )

    transform_configs.append(
        {
            "name": "RandomAffine (full params)",
            "pil": pil_transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
            "cv": cv_transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
        }
    )

    # LinearTransformation is very slow for large matrices, skip for now
    # It would need a much smaller matrix (e.g., 3x3) for reasonable performance

    # Task 11: Random containers
    # RandomApply
    color_jitter = {
        "pil": pil_transforms.ColorJitter(brightness=0.5),
        "cv": cv_transforms.ColorJitter(brightness=0.5),
    }
    transform_configs.append(
        {
            "name": "RandomApply (ColorJitter, p=1.0)",
            "pil": pil_transforms.RandomApply([color_jitter["pil"]], p=1.0),
            "cv": cv_transforms.RandomApply([color_jitter["cv"]], p=1.0),
        }
    )

    # RandomChoice
    transforms_list_pil = [
        pil_transforms.RandomHorizontalFlip(p=1.0),
        pil_transforms.RandomVerticalFlip(p=1.0),
    ]
    transforms_list_cv = [
        cv_transforms.RandomHorizontalFlip(p=1.0),
        cv_transforms.RandomVerticalFlip(p=1.0),
    ]
    transform_configs.append(
        {
            "name": "RandomChoice (HFlip or VFlip)",
            "pil": pil_transforms.RandomChoice(transforms_list_pil),
            "cv": cv_transforms.RandomChoice(transforms_list_cv),
        }
    )

    # RandomOrder
    transform_configs.append(
        {
            "name": "RandomOrder (HFlip, VFlip)",
            "pil": pil_transforms.RandomOrder(transforms_list_pil),
            "cv": cv_transforms.RandomOrder(transforms_list_cv),
        }
    )

    # Task 12: Utility transform (Lambda)
    def simple_brightness(img):
        """Simple lambda function to adjust brightness."""
        if isinstance(img, np.ndarray):
            return np.clip(img * 1.2, 0, 255).astype(img.dtype)
        else:  # PIL Image
            if isinstance(img, PIL.Image.Image):
                enhancer = PIL.ImageEnhance.Brightness(img)
                return enhancer.enhance(1.2)
            return img

    transform_configs.append(
        {
            "name": "Lambda (brightness * 1.2)",
            "pil": pil_transforms.Lambda(simple_brightness),
            "cv": cv_transforms.Lambda(simple_brightness),
        }
    )

    return transform_configs


def main():
    # Load test samples or create synthetic ones
    try:
        samples = load_test_samples(num_samples=50)
        pil_images, cv_images = prepare_images(samples)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Creating synthetic test images instead...")
        # Create synthetic images for testing
        pil_images = [
            Image.fromarray(np.random.randint(0, 255, (500, 375, 3), dtype=np.uint8))
            for _ in range(50)
        ]
        cv_images = [np.array(img) for img in pil_images]

    print(f"Using {len(pil_images)} test images")

    # Get all transform configurations
    transform_configs = get_all_transform_configs()

    print(f"\nTotal transforms to benchmark: {len(transform_configs)}")
    # Test all transforms now

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
            f"{result['pil_avg_time'] * 1000:<12.3f} "
            f"{result['cv_avg_time'] * 1000:<12.3f} "
            f"{result['speedup']:<10.2f}x"
        )

    avg_speedup = np.mean([r["speedup"] for r in results])
    print("-" * 60)
    print(f"{'Average Speedup:':<30} {'':<12} {'':<12} {avg_speedup:<10.2f}x")

    # Plot results
    plot_results(results, save_path="imagenet_benchmark_results.png")


if __name__ == "__main__":
    main()
