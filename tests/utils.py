from typing import Union

import numpy as np
from PIL.Image import Image as PIL_image # for typing


def L1(pil: Union[PIL_image, np.ndarray], np_image: np.ndarray) -> float:
    return np.abs(np.asarray(pil) - np_image).mean()