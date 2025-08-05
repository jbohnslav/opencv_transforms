#!/usr/bin/env python3

import os
import random

import numpy as np
from PIL import Image
from torchvision import transforms as pil_transforms

from opencv_transforms import transforms
from opencv_transforms.debug.utils import compare_random_resized_crop_outputs
from opencv_transforms.debug.utils import debug_random_resized_crop_parameters
from opencv_transforms.debug.utils import test_random_resized_crop_scenarios
from opencv_transforms.debug.visualization import create_random_resized_crop_comparison
from opencv_transforms.debug.visualization import (
    create_random_resized_crop_parameter_comparison,
)


def main():
    """Comprehensive RandomResizedCrop debugging using integrated debug module."""

    print("RandomResizedCrop Debugging with Integrated Debug Module")
    print("=" * 60)

    # Create test image
    height, width = 200, 300
    test_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a pattern that will show differences clearly
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            if (i // 20 + j // 20) % 2 == 0:
                test_image[i : i + 20, j : j + 20] = [255, 255, 255]
            else:
                test_image[i : i + 20, j : j + 20] = [100, 150, 200]

    pil_image = Image.fromarray(test_image)

    # Test basic comparison
    print("\n1. Basic RandomResizedCrop Comparison")
    print("-" * 40)
    result = compare_random_resized_crop_outputs(pil_image, size=224, scale=(0.8, 1.0))

    # Debug parameter generation
    print("\n2. Parameter Generation Analysis")
    print("-" * 40)
    param_debug = debug_random_resized_crop_parameters(
        pil_image, size=224, scale=(0.8, 1.0)
    )

    # Test multiple scenarios
    print("\n3. Multiple Scenario Testing")
    print("-" * 40)
    scenario_results = test_random_resized_crop_scenarios(pil_image)

    # Create output directory
    output_dir = "random_resized_crop_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    print(f"\n4. Generating Visualizations (saved to {output_dir}/)")
    print("-" * 40)

    # Individual comparison visualization
    random.seed(42)
    pil_result = pil_transforms.RandomResizedCrop(224, scale=(0.8, 1.0))(pil_image)
    random.seed(42)
    cv_result = transforms.RandomResizedCrop(224, scale=(0.8, 1.0))(np.array(pil_image))

    params_for_viz = {
        "i": param_debug["main_params"]["i"],
        "j": param_debug["main_params"]["j"],
        "h": param_debug["main_params"]["h"],
        "w": param_debug["main_params"]["w"],
        "scale": "(0.8, 1.0)",
        "size": 224,
    }

    create_random_resized_crop_comparison(
        pil_result, cv_result, params_for_viz, save_path=f"{output_dir}/comparison.png"
    )
    print("✓ Comparison visualization saved")

    # Parameter analysis visualization
    create_random_resized_crop_parameter_comparison(
        param_debug["param_results"], save_path=f"{output_dir}/parameter_analysis.png"
    )
    print("✓ Parameter analysis visualization saved")

    # Summary
    print("\n5. Summary")
    print("-" * 40)
    print(f"Max difference found: {result['max_diff']:.2f}")
    print("Current tolerance (resize): 120.0")
    print(f"Passes tolerance: {'✓' if result['passes_tolerance'] else '✗'}")

    failed_scenarios = [r for r in scenario_results if not r["passes_tolerance"]]
    print(f"Failed scenarios: {len(failed_scenarios)}/{len(scenario_results)}")

    if failed_scenarios:
        print("Failed scenarios:")
        for scenario in failed_scenarios:
            print(
                f"  - {scenario['name']}: {scenario['max_diff']:.1f} (size={scenario['size']}, scale={scenario['scale']})"
            )

    print(f"\nDetailed analysis saved to: {output_dir}/")


if __name__ == "__main__":
    main()
