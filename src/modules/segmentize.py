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
    # Element of result_of_random_walker will be one of below.
    # ignored: -1, background: 1, foreground: 2
    result_of_random_walker: np.ndarray = random_walker(smoothed_img, seed_markers, beta=10, mode='bf')
    # Set ignored as background
    result_of_random_walker[result_of_random_walker == -1] = 1
    # ATTENTION: background label is changed to -1
    # See https://scikit-image.org/docs/0.13.x/api/skimage.measure.html#skimage.measure.label
    labels = label(result_of_random_walker, background=1) - 1
    return labels
