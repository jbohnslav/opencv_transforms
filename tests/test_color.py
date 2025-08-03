import glob
import numpy as np
import random
from typing import Union

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL.Image import Image as PIL_image  # for typing
import pytest
from torchvision import transforms as pil_transforms
from torchvision.transforms import functional as F_pil

from opencv_transforms import transforms
from opencv_transforms import functional as F
from setup_testing_directory import get_testing_directory

TOL = 1e-4

datadir = get_testing_directory()
train_images = glob.glob(datadir + '/**/*.JPEG', recursive=True)
train_images.sort()
print('Number of training images: {:,}'.format(len(train_images)))

random.seed(1)
imfile = random.choice(train_images)
pil_image = Image.open(imfile)
image = cv2.cvtColor(cv2.imread(imfile, 1), cv2.COLOR_BGR2RGB)


class TestContrast:
    @pytest.mark.parametrize('random_seed', [1, 2, 3, 4])
    @pytest.mark.parametrize('contrast_factor', [0.0, 0.5, 1.0, 2.0])
    def test_contrast(self, contrast_factor, random_seed):
        random.seed(random_seed)
        imfile = random.choice(train_images)
        pil_image = Image.open(imfile)
        image = np.array(pil_image).copy()

        pil_enhanced = F_pil.adjust_contrast(pil_image, contrast_factor)
        np_enhanced = F.adjust_contrast(image, contrast_factor)
        assert np.array_equal(np.array(pil_enhanced), np_enhanced.squeeze())

    @pytest.mark.parametrize('n_images', [1, 11])
    def test_multichannel_contrast(self, n_images, contrast_factor=0.1):
        imfile = random.choice(train_images)

        pil_image = Image.open(imfile)
        image = np.array(pil_image).copy()

        multichannel_image = np.concatenate([image for _ in range(n_images)], axis=-1)
        # this will raise an exception in version 0.0.5
        np_enchanced = F.adjust_contrast(multichannel_image, contrast_factor)

    @pytest.mark.parametrize('contrast_factor', [0, 0.5, 1.0])
    def test_grayscale_contrast(self, contrast_factor):
        imfile = random.choice(train_images)

        pil_image = Image.open(imfile)
        image = np.array(pil_image).copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # make sure grayscale images work
        pil_image = pil_image.convert('L')

        pil_enhanced = F_pil.adjust_contrast(pil_image, contrast_factor)
        np_enhanced = F.adjust_contrast(image, contrast_factor)
        assert np.array_equal(np.array(pil_enhanced), np_enhanced.squeeze())
