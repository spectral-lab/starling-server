from unittest import TestCase, skip
from .. import compute_seeds
from .helpers import iter_all_files
import numpy as np
from random import random
import os

__dirname__ = os.path.dirname(os.path.realpath(__file__))


class TestComputeMarks(TestCase):
    # @skip("")
    def test_with_9_items(self):
        mock_array = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        ) * 0.1
        actual_marks = compute_seeds(mock_array, 0.6)
        expected_marks = np.array(
            [[1, 1, 0],
             [0, 0, 0],
             [0, 2, 2]]
        )
        self.assertTrue(np.array_equal(actual_marks, expected_marks))

    # @skip("")
    def test_proportion_of_zero(self):
        mock_array = np.random.rand(100).reshape(10, 10)
        expected_proportion = random()
        marks = compute_seeds(mock_array, expected_proportion)
        actual_proportion = marks[marks == 0].size / marks.size
        self.assertTrue(
            expected_proportion * 0.8 < actual_proportion < expected_proportion * 1.2)

    # @skip("")
    def test_with_real_data(self):
        @iter_all_files(__dirname__ + '/data/spectrogram')
        def check_reproducing_seed_markers(file_path):
            spectrogram_img = np.load(file_path)
            actual_seed_markers = compute_seeds(spectrogram_img)
            seed_markers_path = os.path.join(__dirname__, "data/seed_markers", os.path.basename(file_path))
            expected_seed_markers = np.load(seed_markers_path)
            self.assertTrue(np.array_equal(actual_seed_markers, expected_seed_markers))

        check_reproducing_seed_markers()
