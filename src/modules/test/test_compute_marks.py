from unittest import TestCase, skip
from .. import compute_marks, plot
import numpy as np
from random import random
from pdb import set_trace


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
        self.assertTrue(np.all(actual_marks == expected_marks))

    # @skip("")
    def test_proportion_of_zero(self):
        mock_array = np.random.rand(100).reshape(10, 10)
        expected_proportion = random()
        marks = compute_marks(mock_array, expected_proportion)
        actual_proportion = marks[marks == 0].size / marks.size
        self.assertTrue(
            expected_proportion * 0.9 < actual_proportion < expected_proportion * 1.1)
