import numpy as np
from .plot import plot


def compute_marks(img: np.ndarray, proportion_of_zero: float = 0.999) -> np.ndarray:
    """
    Mark hottest area as 2, coldest area as 1, and the other area as 0.
    """
    is_cold = detect_cold_pixels(img, img.size * (1 - proportion_of_zero) / 2)
    is_hot = detect_hot_pixels(img, img.size * (1 - proportion_of_zero) / 2)

    markers = np.zeros(img.shape, dtype=np.uint)
    markers[is_cold] = 1
    markers[is_hot] = 2
    return markers


def detect_cold_pixels(img: np.ndarray, target_num_pixels: int or float) -> np.array:
    is_cold = np.zeros(img.shape, dtype=bool)
    num_detected = 0
    unique, counts = np.unique(img, return_counts=True)
    while target_num_pixels > num_detected:
        min_magnitude = unique.min()
        is_cold[img == min_magnitude] = True
        num_detected += counts[unique == min_magnitude][0]
        unique[unique == min_magnitude] = 1.
    return is_cold


def detect_hot_pixels(img: np.ndarray, target_num_pixels: int or float) -> np.array:
    is_hot = np.zeros(img.shape, dtype=bool)
    num_detected = 0
    unique, counts = np.unique(img, return_counts=True)
    while target_num_pixels > num_detected:
        max_magnitude = unique.max()
        is_hot[img == max_magnitude] = True
        num_detected += counts[unique == max_magnitude][0]
        unique[unique == max_magnitude] = 0.
    return is_hot
