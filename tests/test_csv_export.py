#!/usr/bin/env python3
"""Test script for CSV export functionality."""

import sys

sys.path.append(".")

import os

import pandas as pd

from benchmark import export_results_to_csv
from benchmark import get_transform_configs
from benchmark import load_cityscapes_dataset
from benchmark import load_imagenet_validation
from benchmark import prepare_multi_size_images
from benchmark import run_benchmark


def test_csv_export():
    """Test the CSV export functionality."""
    print("=" * 60)
    print("TESTING CSV EXPORT")
    print("=" * 60)

    # Load small samples for speed
    print("\n1. Loading datasets...")
    cityscapes_samples = load_cityscapes_dataset(num_samples=2)
    imagenet_samples = load_imagenet_validation(num_samples=2)

    # Prepare multi-size images (just 2 sizes for testing)
    print("\n2. Preparing multi-size images...")
    size_dict = prepare_multi_size_images(cityscapes_samples, imagenet_samples)
    test_sizes = [(1024, 512), (224, 224)]  # Test 2 sizes

    # Get subset of transforms for testing
    print("\n3. Getting transform configs...")
    all_configs = get_transform_configs()
    test_configs = all_configs[:3]  # Test first 3 transforms

    print(f"Testing {len(test_configs)} transforms across {len(test_sizes)} sizes")

    # Generate test results
    print("\n4. Generating test results...")
    results = []

    for size in test_sizes:
        if size not in size_dict:
            continue

        pil_images, cv_images = size_dict[size]

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

    # Test CSV export
    print("\n6. Testing CSV export...")
    test_csv_path = "test_benchmark_results.csv"

    try:
        export_results_to_csv(results, csv_path=test_csv_path)
        print("   ✅ CSV export successful")

        # Verify the CSV file was created and has correct structure
        if os.path.exists(test_csv_path):
            df = pd.read_csv(test_csv_path)
            print(
                f"   ✅ CSV file created with {len(df)} rows, {len(df.columns)} columns"
            )
            print(f"   Columns: {list(df.columns)}")

            # Verify expected columns exist
            expected_cols = [
                "transform",
                "image_size",
                "width",
                "height",
                "pixel_count",
                "pil_avg_time_ms",
                "cv_avg_time_ms",
                "speedup",
                "timestamp",
                "system_info",
            ]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                print(f"   ❌ Missing columns: {missing_cols}")
            else:
                print("   ✅ All expected columns present")

            # Show sample data
            print("\n   Sample data:")
            print(df[["transform", "image_size", "speedup"]].head())

        else:
            print("   ❌ CSV file was not created")

    except Exception as e:
        print(f"   ❌ CSV export failed: {e}")

    # Clean up test file
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
        print(f"\n   🧹 Cleaned up {test_csv_path}")

    print("\n✅ CSV export test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_csv_export()
