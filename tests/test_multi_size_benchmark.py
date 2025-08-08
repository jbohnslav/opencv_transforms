#!/usr/bin/env python3
"""Quick test script to validate multi-size benchmarking works."""

import sys

sys.path.append(".")

from benchmark import get_transform_configs
from benchmark import load_cityscapes_dataset
from benchmark import load_imagenet_validation
from benchmark import prepare_multi_size_images
from benchmark import run_benchmark


def test_multi_size_benchmark():
    """Test the updated multi-size benchmark functions."""
    print("=" * 60)
    print("TESTING MULTI-SIZE BENCHMARK")
    print("=" * 60)

    # Load small samples for speed
    print("\n1. Loading datasets...")
    cityscapes_samples = load_cityscapes_dataset(num_samples=3)
    imagenet_samples = load_imagenet_validation(num_samples=3)

    # Prepare multi-size images
    print("\n2. Preparing multi-size images...")
    size_dict = prepare_multi_size_images(cityscapes_samples, imagenet_samples)

    # Get a subset of transforms for testing
    print("\n3. Getting transform configs...")
    all_configs = get_transform_configs()
    test_configs = all_configs[:3]  # Test first 3 transforms only

    print(f"Testing {len(test_configs)} transforms:")
    for config in test_configs:
        print(f"   - {config['name']}")

    # Test benchmark on different sizes
    print("\n4. Running benchmark tests...")
    sizes = sorted(size_dict.keys(), key=lambda x: x[0] * x[1], reverse=True)

    results = []
    for size in sizes[:2]:  # Test first 2 sizes only
        width, height = size
        pil_images, cv_images = size_dict[size]
        print(f"\nTesting size {width}x{height}...")

        for config in test_configs:
            try:
                result = run_benchmark(
                    config["name"],
                    config["pil"],
                    config["cv"],
                    pil_images,
                    cv_images,
                    size,
                )
                results.append(result)
                print(f"   ✅ {config['name']}: {result['speedup']:.2f}x speedup")
            except Exception as e:
                print(f"   ❌ {config['name']}: Error - {e}")

    print("\n5. Summary:")
    print(f"   - Tested {len(results)} transform/size combinations")
    print(
        f"   - All results include size information: {all('image_size' in r for r in results)}"
    )
    print(
        f"   - Example result keys: {list(results[0].keys()) if results else 'No results'}"
    )

    print("\n✅ Multi-size benchmark test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_multi_size_benchmark()
