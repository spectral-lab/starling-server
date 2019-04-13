from skimage.segmentation import random_walker
from skimage.filters import gaussian
from skimage.measure import label
import numpy as np

def segmentize(img: np.ndarray, line_continuity: int, peak_level: float, background_level: float) -> np.ndarray:
    smoothed_img = gaussian(img, sigma=(line_continuity * 2, line_continuity * 0.1))
    markers = np.zeros(smoothed_img.shape, dtype=np.uint)
    markers[smoothed_img < background_level] = 1
    markers[smoothed_img > peak_level] = 2
    binary_segment = random_walker(smoothed_img, markers, beta=10, mode='bf') - 1
    labels = label(np.array(binary_segment)) - 1
    return labels
