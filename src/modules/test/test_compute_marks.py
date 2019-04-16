from unittest import TestCase, skip
from .. import compute_marks, plot
from .helpers import iter_all_files
import numpy as np
from random import random
from pdb import set_trace
import os

__dir__ = os.path.dirname(os.path.realpath(__file__))


class TestComputeMarks(TestCase):
    # @skip("")
    def test_with_9_items(self):
        mock_array = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        ) * 0.1
        actual_marks = compute_marks(mock_array, 0.6)
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
        marks = compute_marks(mock_array, expected_proportion)
        actual_proportion = marks[marks == 0].size / marks.size
        self.assertTrue(
            expected_proportion * 0.8 < actual_proportion < expected_proportion * 1.2)

    # @skip("")
    def test_with_real_data(self):
        @iter_all_files(__dir__ + '/data/img_from_client')
        def check_reproducing_markers(file_path):
            img = np.load(file_path)
            actual_markers = compute_marks(img)
            markers_path = os.path.join(__dir__, "data/markers", os.path.basename(file_path))
            expected_markers = np.load(markers_path)
            self.assertTrue(np.array_equal(actual_markers, expected_markers))

        check_reproducing_markers()

    # @skip("Takes a lot of time to generate graphs")
    def test_plotting(self):
        @iter_all_files(__dir__ + '/data/img_from_client')
        def plot_markers(file_path):
            img = np.load(file_path)
            markers = compute_marks(img)
            title = os.path.splitext(os.path.basename(file_path))[0]
            plot(markers, title)

        plot_markers()
