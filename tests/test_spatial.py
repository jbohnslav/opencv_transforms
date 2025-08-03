import glob
import numpy as np
import random
from typing import Union

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL.Image import Image as PIL_image # for typing

from torchvision import transforms as pil_transforms
from torchvision.transforms import functional as F_pil
from opencv_transforms import transforms
from opencv_transforms import functional as F

from setup_testing_directory import get_testing_directory
from utils import L1

TOL = 1e-4

datadir = get_testing_directory()
train_images = glob.glob(datadir + '/**/*.JPEG', recursive=True)
train_images.sort()
print('Number of training images: {:,}'.format(len(train_images)))

random.seed(1)
imfile = random.choice(train_images)
pil_image = Image.open(imfile)
image = cv2.cvtColor(cv2.imread(imfile, 1), cv2.COLOR_BGR2RGB)


def test_resize():
    pil_resized = pil_transforms.Resize((224, 224))(pil_image)
    resized = transforms.Resize((224, 224))(image)
    l1 = L1(pil_resized, resized)
    assert l1 - 88.9559 < TOL

def test_rotation():
    random.seed(1)
    pil = pil_transforms.RandomRotation(10)(pil_image)
    random.seed(1)
    np_img = transforms.RandomRotation(10)(image)
    l1 = L1(pil, np_img)
    assert l1 - 86.7955 < TOL

def test_five_crop():
    pil = pil_transforms.FiveCrop((224, 224))(pil_image)
    cv = transforms.FiveCrop((224, 224))(image)
    pil_stacked = np.hstack([np.asarray(i) for i in pil])
    cv_stacked = np.hstack(cv)
    l1 = L1(pil_stacked, cv_stacked)
    assert l1 - 22.0444 < TOL