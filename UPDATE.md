# Missing Transforms from Latest PyTorch Vision

This document analyzes transforms available in the latest torchvision (from https://github.com/pytorch/vision/tree/main/torchvision/transforms) that are not implemented in opencv_transforms.

## Current opencv_transforms Status

The opencv_transforms library currently implements 24 transforms:
- CenterCrop, ColorJitter, Compose, FiveCrop, Grayscale, Lambda, LinearTransformation
- Normalize, Pad, RandomAffine, RandomApply, RandomChoice, RandomCrop, RandomGrayscale
- RandomHorizontalFlip, RandomOrder, RandomResizedCrop, RandomRotation, RandomSizedCrop
- RandomVerticalFlip, Resize, Scale, TenCrop, ToTensor

## Missing Core Transforms

The following transforms exist in torchvision but are **missing** from opencv_transforms:

### 1. Image Type Conversion Transforms
- **PILToTensor** - Converts PIL Image to tensor without scaling to [0, 1]
- **ConvertImageDtype** - Converts tensor image to different dtype
- **ToPILImage** - Converts tensor/ndarray to PIL Image

### 2. Perspective & Geometric Transforms
- **RandomPerspective** - Performs random perspective transformation

### 3. Advanced Color/Pixel Transforms
- **RandomInvert** - Inverts colors of the image randomly
- **RandomPosterize** - Reduces number of bits for each color channel
- **RandomSolarize** - Inverts all pixel values above a threshold
- **RandomAdjustSharpness** - Adjusts sharpness of the image
- **RandomAutocontrast** - Applies autocontrast to the image
- **RandomEqualize** - Equalizes the histogram of the image

### 4. Image Enhancement Transforms
- **GaussianBlur** - Applies Gaussian blur to the image
- **ElasticTransform** - Applies elastic transformation (distortion)

### 5. Augmentation Transforms
- **RandomErasing** - Randomly erases rectangular regions in the image

### 6. Auto-Augmentation Transforms (from autoaugment.py)
- **AutoAugment** - Automatically selects best augmentation policies
- **RandAugment** - Random augmentation with reduced search space
- **TrivialAugmentWide** - Simple, tuning-free augmentation
- **AugMix** - Augmentation mixing for improved robustness

## Priority Recommendations

### High Priority (Common Use Cases)
1. **RandomPerspective** - Useful for data augmentation
2. **GaussianBlur** - Common preprocessing/augmentation
3. **RandomErasing** - Popular augmentation technique
4. **PILToTensor** - Important for compatibility

### Medium Priority (Specialized but Useful)
5. **ElasticTransform** - Useful for certain domains (medical imaging)
6. **RandomAutocontrast** - Useful color augmentation
7. **RandomEqualize** - Histogram equalization
8. **ConvertImageDtype** - Type conversion utility
9. **ToPILImage** - Conversion utility

### Lower Priority (Less Common)
10. **RandomInvert** - Specific use cases
11. **RandomPosterize** - Artistic effect
12. **RandomSolarize** - Specific augmentation
13. **RandomAdjustSharpness** - Less common augmentation

### Advanced/Complex (Requires Significant Implementation)
14. **AutoAugment** - Complex policy-based system
15. **RandAugment** - Requires augmentation search
16. **TrivialAugmentWide** - Requires specific implementation
17. **AugMix** - Complex mixing strategy

## Implementation Notes

1. **Compatibility Focus**: When implementing these transforms, ensure they match torchvision's behavior exactly, as per the testing philosophy in TEST_PLAN.md

2. **OpenCV Equivalents**: Many of these can be implemented using OpenCV functions:
   - GaussianBlur → cv2.GaussianBlur
   - RandomPerspective → cv2.getPerspectiveTransform + cv2.warpPerspective
   - ElasticTransform → cv2.remap with custom displacement fields

3. **Type Conversions**: PILToTensor, ConvertImageDtype, and ToPILImage require careful handling of data types and formats

4. **Auto-Augmentation**: The auto-augmentation transforms (AutoAugment, RandAugment, etc.) are complex systems that would require significant implementation effort and might be better left for a future major version

## Summary

opencv_transforms is missing **17 core transforms** and **4 auto-augmentation systems** compared to the latest torchvision. The highest priority additions would be:
- RandomPerspective
- GaussianBlur
- RandomErasing
- PILToTensor
- ElasticTransform

These would significantly improve feature parity with torchvision while maintaining the performance benefits of OpenCV-based implementations.