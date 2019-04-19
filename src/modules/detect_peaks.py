import numpy as np
from typing import List


def detect_peaks(spectrogram2d: np.ndarray, segment_labels: np.ndarray) -> List[List[List[int]]]:
    """
    Detect Peaks from segments of spectrogram.

    :param spectrogram2d:
    :param segment_labels: labels which indicates background as -1 and foreground as index starts from 0
    :return: [row, column] pairs splitted into some chunks which represents a peak line.
    """
    peak_points = []
    for target_label in range(segment_labels.max() + 1):
        target_area: np.ndarray = segment_labels == target_label
        one_chunk_of_points = find_peaks_in_segment(spectrogram2d, target_area)
        peak_points.append(one_chunk_of_points)
    return peak_points


def find_peaks_in_segment(spectrogram2d: np.ndarray, target_area: np.ndarray) -> List[List[int]]:
    i, j = np.where(target_area)
    segment_columns = np.unique(j)
    one_chunk_of_points = []
    for column_idx in segment_columns:
        current_column = spectrogram2d[:, column_idx]
        is_segment = target_area[:, column_idx]
        max_val = current_column[is_segment].max()
        row_idx = np.where(current_column == max_val)[0][0]
        one_chunk_of_points.append([int(row_idx), int(column_idx)])
    return one_chunk_of_points
