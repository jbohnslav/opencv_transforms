# opencv_transforms

This repository is intended as a faster drop-in replacement for [Pytorch's Torchvision augmentations](https://github.com/pytorch/vision/). Instead of relying on Pillow for image augmentation, this repository requires OpenCV.

## Requirements
* A working installation of OpenCV. **Tested with OpenCV version 3.4.1**
* Tested on Windows 10.

## Installation
* `git clone https://github.com/jbohnslav/opencv_transforms.git`
* Add to your python path

## Usage
* `from opencv_transforms import opencv_transforms as transforms`
* From here, almost everything should work exactly as the original `transforms`.
* Examples: 
** `transformed_image = transforms.Resize(size=(256,256))(image)`
** Should be 1.5 to 10 times faster than PIL, depending on settings.

## Performance
* An example benchmarking file can be found in the notebook **bencharming_v2.ipynb**
* Most transformations are between 1.5X and ~4X faster in OpenCV.
![resize](benchmarks/benchmarking_Resize.png)
![random crop](benchmarks/benchmarking_Random crop quarter size)