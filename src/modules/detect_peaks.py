import numpy as np
from typing import List

"""
Detect Peaks in segments of spectrogram. 
Return List is [row, column] pairs splitted into some chunks which represents a peak line.
"""


def detect_peaks(spectrogram2d: np.ndarray, labels: np.ndarray) -> List[List[List[int]]]:
    peak_points = []
    for i in range(labels.max()):
        segment = labels == i
        one_chunk_of_points = find_peaks_in_segment(spectrogram2d, segment)
        peak_points.append(one_chunk_of_points)
    return peak_points


def find_peaks_in_segment(spectrogram2d: np.ndarray, segment: np.ndarray) -> List[List[int]]:
    i, j = np.where(segment)
    segment_columns = np.unique(j)
    one_chunk_of_points = []
    for column_idx in segment_columns:
        current_column = spectrogram2d[:, column_idx]
        is_segment = segment[:, column_idx]
        max_val = current_column[is_segment].max()
        row_idx = np.where(current_column == max_val)[0][0]
        one_chunk_of_points.append([int(row_idx), int(column_idx)])
    return one_chunk_of_points
