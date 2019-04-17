from skimage.segmentation import random_walker
from skimage.filters import gaussian
from skimage.measure import label
import numpy as np
from pdb import set_trace


def segmentize(img: np.ndarray, seed_markers: np.ndarray, line_continuity: int = 0) -> np.ndarray:
    """
    Segmentize spectrogram image. return labels which indicates background as -1 and foreground as index starts from 0

    :return
        : np.ndarray
            labels which indicates background as -1 and foreground as index starts from 0
    """
    smoothed_img = gaussian(img, sigma=(line_continuity, 0))
    binary_segment = random_walker(smoothed_img, seed_markers, beta=10, mode='bf') - 1
    labels = label(np.array(binary_segment)) - 1
    return labels
