import numpy as np
from typing import List
from math import floor
from .export_graph import export_3d_scatter


def extract_training_points(segment_labels: np.ndarray, spectrogram2d: np.ndarray, ratio=0.5) -> List[np.ndarray]:
    """
    :param segment_labels
    :param spectrogram2d
    :param ratio: ratio to extract training points.
    :return: Each element represents training points in one segment.
             Training points are formatted like [time, freq, magnitude].
             The shape of ndarray is (n, 3).
    """
    if spectrogram2d.shape != segment_labels.shape:
        raise Exception('Input shape does not match')

    list_of_points = make_list_of_points(segment_labels, spectrogram2d)
    export_3d_scatter(list_of_points[0], "before_selection")
    training_points = select_high_magnitude(list_of_points, ratio)
    return training_points


def make_list_of_points(segment_labels, spectrogram2d) -> List[np.ndarray]:
    """
    :return: List of ndarray. Each ndarray represents all the points in the corresponding segment.
    """
    list_of_points = []
    for target_label in range(segment_labels.max() + 1):
        time_indices, freq_indices = np.meshgrid(np.arange(segment_labels.shape[1]), np.arange(segment_labels.shape[0]))
        target_times = time_indices[segment_labels == target_label]
        target_freqs = freq_indices[segment_labels == target_label]
        target_magnitudes = spectrogram2d[segment_labels == target_label]
        list_of_points.append(np.column_stack([target_times, target_freqs, target_magnitudes]))
    return list_of_points


def select_high_magnitude(points, ratio=0.5):
    """
    After determining the threshold based on the ratio param,
    make a list of points which have a higher magnitude than the threshold.
    :param points:
    :param ratio: ratio to select.
    :return:
    """
    selected_points = []
    for i in range(len(points)):
        target_points = points[i]
        num_points_to_select = floor(target_points.shape[0] * ratio)
        magnitudes = target_points[:, 2]
        threshold = np.sort(magnitudes)[-num_points_to_select-1]
        selected_points.append(target_points[magnitudes > threshold, :])
    return selected_points
