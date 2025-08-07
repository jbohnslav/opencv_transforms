# Comprehensive List of All Transforms

## Class-Based Transforms (opencv_transforms.transforms)

### Basic Transforms
1. **Compose** - Composes several transforms together
2. **ToTensor** - Convert PIL/numpy to tensor
3. **Normalize** - Normalize tensor with mean and std

### Resizing Transforms  
4. **Resize** - Resize image to given size
5. **Scale** - Alias for Resize (deprecated)
6. **RandomResizedCrop** - Crop random portion and resize
7. **RandomSizedCrop** - Alias for RandomResizedCrop (deprecated)

### Cropping Transforms
8. **CenterCrop** - Crop center of image
9. **RandomCrop** - Crop random portion
10. **FiveCrop** - Crop image into 5 patches (corners + center)
11. **TenCrop** - Crop image into 10 patches (FiveCrop + flipped)

### Flipping Transforms
12. **RandomHorizontalFlip** - Random horizontal flip
13. **RandomVerticalFlip** - Random vertical flip

### Padding Transform
14. **Pad** - Pad image on all sides

### Color Transforms
15. **ColorJitter** - Random brightness, contrast, saturation, hue
16. **Grayscale** - Convert to grayscale
17. **RandomGrayscale** - Random grayscale conversion

### Geometric Transforms
18. **RandomRotation** - Random rotation
19. **RandomAffine** - Random affine transformation
20. **LinearTransformation** - Linear transformation with matrix

### Random Containers
21. **RandomApply** - Apply transforms with probability
22. **RandomChoice** - Select one random transform
23. **RandomOrder** - Apply transforms in random order

### Utility Transform
24. **Lambda** - Apply user-defined lambda

## Functional Transforms (opencv_transforms.functional)

### Basic Functions
25. **to_tensor** - Convert to tensor
26. **normalize** - Normalize tensor
27. **pad** - Pad image

### Resizing Functions
28. **resize** - Resize image
29. **scale** - Alias for resize (deprecated)
30. **resized_crop** - Crop and resize

### Cropping Functions
31. **crop** - Crop image at specific location
32. **center_crop** - Crop center
33. **five_crop** - Create 5 crops
34. **ten_crop** - Create 10 crops

### Flipping Functions
35. **hflip** - Horizontal flip
36. **vflip** - Vertical flip

### Color Adjustment Functions
37. **adjust_brightness** - Adjust brightness
38. **adjust_contrast** - Adjust contrast
39. **adjust_saturation** - Adjust saturation
40. **adjust_hue** - Adjust hue
41. **adjust_gamma** - Adjust gamma

### Geometric Functions
42. **rotate** - Rotate image
43. **affine** - Apply affine transformation

### Grayscale Function
44. **to_grayscale** - Convert to grayscale

## Total Count
- **24 Class-based transforms**
- **20 Functional transforms**
- **44 Total transforms**

## Notes for Benchmarking
- Some transforms require parameters (e.g., size, degrees)
- Random transforms need fixed seed for reproducibility
- Some transforms are aliases/deprecated but should still be tested
- Functional transforms need wrapper for class-like interface