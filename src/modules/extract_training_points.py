import numpy as np
from typing import List
from math import floor, ceil
from .export_graph import export_3d_scatter
from pdb import set_trace


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

    selected_points = []
    for target_label in range(segment_labels.max() + 1):
        masked_spectrogram = spectrogram2d * (segment_labels == target_label)
        time_indices, freq_indices = np.meshgrid(np.arange(segment_labels.shape[1]), np.arange(segment_labels.shape[0]))

        should_select = audition_magnitude(masked_spectrogram, ratio)

        selected_points.append(
            np.column_stack([
                time_indices[should_select],
                freq_indices[should_select],
                spectrogram2d[should_select]
            ])
        )

    return selected_points


def audition_magnitude(masked_spectrogram, ratio=0.5) -> np.ndarray:
    """
    After determining the threshold based on the ratio param,
    make ndarray of bool which indicates if the magnitude is higher or not
    :param masked_spectrogram:
    :param ratio: ratio to select.
    """
    times_to_check = np.unique(np.nonzero(masked_spectrogram)[1])
    is_higher = np.zeros(masked_spectrogram.shape, dtype=bool)

    for time_idx in times_to_check:
        column_of_spectrogram = masked_spectrogram[:, time_idx]
        magnitudes = np.unique(column_of_spectrogram)
        num_nonzero = np.count_nonzero(magnitudes)
        num_points_to_pass = ceil(num_nonzero * ratio)

        # TODO: should write a test
        # May cause problem
        threshold = 0
        if num_nonzero > 1:
            threshold = np.sort(magnitudes)[-num_points_to_pass]
        is_higher[column_of_spectrogram >= threshold, time_idx] = True

    return is_higher
