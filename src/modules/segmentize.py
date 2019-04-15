from skimage.segmentation import random_walker
from skimage.filters import gaussian
from skimage.measure import label
import numpy as np
from .compute_marks import compute_marks


def segmentize(img: np.ndarray, line_continuity: int, peak_level: float, background_level: float) -> np.ndarray:
    """
    Segmentize spectrogram image. return labels which indicates background as -1 and foreground as index starts from 0
    """
    smoothed_img = gaussian(img, sigma=(line_continuity * 1, line_continuity * 0.1))
    markers = compute_marks(smoothed_img)
    binary_segment = random_walker(smoothed_img, markers, beta=10, mode='bf') - 1
    labels = label(np.array(binary_segment)) - 1
    return labels
