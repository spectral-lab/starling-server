from unittest import TestCase, skip
from .. import detect_peaks
from typing import List
from random import randint
from .helpers import iter_all_files, flatten_list
import os
import numpy as np
from pdb import set_trace

__dirname__ = os.path.dirname(os.path.realpath(__file__))


class TestDetectPeaks(TestCase):
    @skip("")
    def test_not_empty(self):
        spectrogram2d = np.load(__dirname__ + '/data/spectrogram2d/' + str(randint(0, 29)) + '.npy')
        labels = np.load(__dirname__ + '/data/labels/' + str(randint(0, 29)) + '.npy')
        peak_points = detect_peaks(spectrogram2d, labels)
        peak_points = flatten_list(peak_points)
        self.assertTrue(len(peak_points) > 0)

    @skip("")
    def test_type_check(self):
        spectrogram2d = np.load(__dirname__ + '/data/spectrogram2d/' + str(randint(0, 29)) + '.npy')
        labels = np.load(__dirname__ + '/data/labels/' + str(randint(0, 29)) + '.npy')
        peak_points = detect_peaks(spectrogram2d, labels)
        self.assertIsInstance(peak_points, List)
        self.assertIsInstance(peak_points[0], List)
        self.assertIsInstance(peak_points[0][0], List)
        self.assertIsInstance(peak_points[0][0][0], int)
        for list_of_list in peak_points:
            for a_list in list_of_list:
                for item in a_list:
                    self.assertIsInstance(item, int)

    # @skip("")
    def test_with_real_data(self):
        @iter_all_files(__dirname__ + '/data/labels')
        def check_reproducing_peak_points(file_path):
            spectrogram_img = np.load(os.path.join(__dirname__, 'data/spectrogram', os.path.basename(file_path)))
            labels = np.load(file_path)
            actual_peak_points = detect_peaks(spectrogram_img, labels)
            # np.save(os.path.join(__dirname__, "data/peak_points", os.path.basename(file_path)), actual_peak_points)
            expected_peak_points = np.load(os.path.join(__dirname__, "data/peak_points", os.path.basename(file_path)))
            self.assertTrue(np.array_equal(actual_peak_points, expected_peak_points))

        check_reproducing_peak_points()
