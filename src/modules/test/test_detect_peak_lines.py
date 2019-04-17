from unittest import TestCase, skip
from .. import detect_peaks, export_graph, format_as_2d_array
from typing import List
from .helpers import iter_all_files
import os
import numpy as np
from pdb import set_trace

__dirname__ = os.path.dirname(os.path.realpath(__file__))


class TestDetectPeakLines(TestCase):
    # @skip("")
    def test_with_real_data(self):
        # @iter_all_files(__dirname__ + '/data/labels')
        def check_reproducing_peak_lines(file_path):
            spectrogram_img = np.load(os.path.join(__dirname__, 'data/spectrogram', os.path.basename(file_path)))
            labels = np.load(file_path)
            actual_peak_lines = detect_peaks(spectrogram_img, labels)
            np.save(os.path.join(__dirname__, "data/peak_lines", os.path.basename(file_path)), actual_peak_lines)
            title = os.path.splitext(os.path.basename(file_path))[0]
            export_graph(format_as_2d_array(actual_peak_lines, spectrogram_img.shape), "peak_points_" + title)

            # expected_peak_points = np.load(os.path.join(__dirname__, "data/peak_points", os.path.basename(file_path)))
            # self.assertTrue(np.array_equal(actual_peak_points, expected_peak_points))

        check_reproducing_peak_lines(__dirname__ + '/data/labels/bird.npy')
