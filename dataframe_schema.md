# Benchmark Results DataFrame Schema

## Primary DataFrame Columns

### Core Identification
- **transform_name** (str): Name of the transform (e.g., "Resize", "ColorJitter")
- **transform_type** (str): Type of transform ("class" or "functional")
- **parameters** (str): Parameters used for the transform (e.g., "size=(256,256)")

### Timing Metrics
- **library** (str): Library used ("PIL/torchvision" or "OpenCV")
- **avg_time_ms** (float): Average time per image in milliseconds
- **std_time_ms** (float): Standard deviation of time per image
- **total_time_s** (float): Total time for all images in seconds
- **min_time_ms** (float): Minimum time for single image
- **max_time_ms** (float): Maximum time for single image

### Performance Comparison
- **speedup_ratio** (float): Speedup of OpenCV over PIL (PIL_time / OpenCV_time)
- **relative_performance** (str): Categorical ("faster", "similar", "slower")

### Test Configuration
- **num_images** (int): Number of images tested
- **avg_image_size** (str): Average input image dimensions (e.g., "500x500")
- **date_tested** (datetime): Timestamp of when test was run
- **system_info** (str): System/hardware info (optional)

## Example DataFrame Structure

```python
import pandas as pd

# Example row
{
    'transform_name': 'Resize',
    'transform_type': 'class',
    'parameters': 'size=(256,256)',
    'library': 'OpenCV',
    'avg_time_ms': 0.523,
    'std_time_ms': 0.082,
    'total_time_s': 0.0523,
    'min_time_ms': 0.412,
    'max_time_ms': 0.891,
    'speedup_ratio': 2.34,
    'relative_performance': 'faster',
    'num_images': 100,
    'avg_image_size': '500x375',
    'date_tested': '2024-01-15 14:30:00',
    'system_info': 'MacOS M1'
}
```

## Aggregated Statistics DataFrame

### Summary Metrics
- **metric_name** (str): Name of the metric
- **value** (float/str): Value of the metric

### Key Metrics to Track:
1. **overall_avg_speedup**: Average speedup across all transforms
2. **median_speedup**: Median speedup value
3. **fastest_transform**: Transform with highest speedup
4. **fastest_speedup**: Value of highest speedup
5. **slowest_transform**: Transform with lowest speedup
6. **slowest_speedup**: Value of lowest speedup
7. **transforms_faster**: Count of transforms where OpenCV is faster
8. **transforms_slower**: Count of transforms where OpenCV is slower
9. **transforms_similar**: Count of transforms with similar performance
10. **total_transforms_tested**: Total number of transforms benchmarked

## CSV Export Structure

### Main Results File: `benchmark_results.csv`
- Contains all rows from primary DataFrame
- One row per transform per library (2 rows per transform)
- Sorted by transform_name, then library

### Summary File: `benchmark_summary.csv`
- Contains aggregated statistics
- Key performance indicators
- System information

### Comparison File: `benchmark_comparison.csv`
- Pivoted view with transforms as rows
- Columns for PIL time, OpenCV time, speedup ratio
- Easier for quick comparison

## Usage in Code

```python
# Create main results DataFrame
results_df = pd.DataFrame(columns=[
    'transform_name', 'transform_type', 'parameters',
    'library', 'avg_time_ms', 'std_time_ms', 'total_time_s',
    'min_time_ms', 'max_time_ms', 'speedup_ratio',
    'relative_performance', 'num_images', 'avg_image_size',
    'date_tested', 'system_info'
])

# After benchmarking, export to CSV
results_df.to_csv('benchmark_results.csv', index=False)
```