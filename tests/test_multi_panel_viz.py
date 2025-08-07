#!/usr/bin/env python3
"""Test script for multi-panel visualization functions."""

import sys

sys.path.append(".")

import matplotlib

matplotlib.use("Agg")  # Use headless backend

from benchmark import get_transform_configs
from benchmark import load_cityscapes_dataset
from benchmark import load_imagenet_validation
from benchmark import plot_multi_size_results
from benchmark import plot_results
from benchmark import plot_speedup_summary
from benchmark import prepare_multi_size_images
from benchmark import run_benchmark


def test_visualization():
    """Test the new multi-panel visualization functions."""
    print("=" * 60)
    print("TESTING MULTI-PANEL VISUALIZATION")
    print("=" * 60)

    # Load small samples for speed
    print("\n1. Loading datasets...")
    cityscapes_samples = load_cityscapes_dataset(num_samples=2)
    imagenet_samples = load_imagenet_validation(num_samples=2)

    # Prepare multi-size images (just 3 sizes for testing)
    print("\n2. Preparing multi-size images...")
    size_dict = prepare_multi_size_images(cityscapes_samples, imagenet_samples)
    test_sizes = [(2048, 1024), (512, 256), (224, 224)]  # Test 3 sizes

    # Get subset of transforms for testing
    print("\n3. Getting transform configs...")
    all_configs = get_transform_configs()
    test_configs = all_configs[:4]  # Test first 4 transforms

    print(f"Testing {len(test_configs)} transforms across {len(test_sizes)} sizes:")
    for config in test_configs:
        print(f"   - {config['name']}")

    # Generate test results
    print("\n4. Generating test results...")
    results = []

    for size in test_sizes:
        if size not in size_dict:
            continue

        pil_images, cv_images = size_dict[size]
        print(f"Processing size {size[0]}x{size[1]}...")

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
            except Exception as e:
                print(f"   ⚠️  {config['name']}: {e}")

    print(f"\n5. Generated {len(results)} test results")

    # Test visualization functions
    print("\n6. Testing visualization functions...")

    try:
        print("   Testing plot_multi_size_results()...")
        plot_multi_size_results(results, save_path="test_multi_panel.png")
        print("   ✅ Multi-panel plot created successfully")
    except Exception as e:
        print(f"   ❌ Multi-panel plot failed: {e}")

    try:
        print("   Testing plot_speedup_summary()...")
        plot_speedup_summary(results, save_path="test_heatmap.png")
        print("   ✅ Speedup heatmap created successfully")
    except Exception as e:
        print(f"   ❌ Speedup heatmap failed: {e}")

    try:
        print("   Testing plot_results() (combined)...")
        plot_results(results, save_path="test_combined_viz.png")
        print("   ✅ Combined visualization created successfully")
    except Exception as e:
        print(f"   ❌ Combined visualization failed: {e}")

    print("\n✅ Multi-panel visualization test complete!")
    print("=" * 60)
    print("Check the generated PNG files to verify visualization quality.")


if __name__ == "__main__":
    test_visualization()
