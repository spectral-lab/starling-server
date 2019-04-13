from unittest import TestCase
from .. import detect_peaks
from typing import List, Tuple
from random import randint
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


def flatten(x):
    return [z for y in x for z in (flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]


class TestDetectPeaks(TestCase):
    def test_not_empty(self):
        spectrogram2d = np.load(dir_path + '/data/spectrogram2d/' + str(randint(0, 29)) + '.npy')
        labels = np.load(dir_path + '/data/labels/' + str(randint(0, 29)) + '.npy')
        peak_points = detect_peaks(spectrogram2d, labels)
        peak_points = flatten(peak_points)
        self.assertTrue(len(peak_points) > 0)

    def test_type_check(self):
        spectrogram2d = np.load(dir_path + '/data/spectrogram2d/' + str(randint(0, 29)) + '.npy')
        labels = np.load(dir_path + '/data/labels/' + str(randint(0, 29)) + '.npy')
        peak_points = detect_peaks(spectrogram2d, labels)
        self.assertIsInstance(peak_points, List)
        self.assertIsInstance(peak_points[0], List)
        self.assertIsInstance(peak_points[0][0], List)
        self.assertIsInstance(peak_points[0][0][0], int)
        for list_of_list in peak_points:
            for list in list_of_list:
                for item in list:
                    self.assertIsInstance(item, int)
