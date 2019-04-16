from unittest import TestCase, skip
from .. import plot
import numpy as np
import os
from .helpers import iter_all_files

__dir__ = os.path.dirname(os.path.realpath(__file__))


# noinspection PyTypeChecker
class TestPlot(TestCase):
    # @skip("Takes a lot of time to generate graphs")
    def test_plot_spectrogram(self):
        @iter_all_files(__dir__ + '/data/spectrogram')
        def plot_spectrogram(file_path):
            spectrogram_img = np.load(file_path)
            title = os.path.splitext(os.path.basename(file_path))[0]
            plot(spectrogram_img, "spectrogram_" + title)

        plot_spectrogram()

    # @skip("Takes a lot of time to generate graphs")
    def test_plot_marks(self):
        @iter_all_files(__dir__ + '/data/markers')
        def plot_markers(file_path):
            markers = np.load(file_path)
            title = os.path.splitext(os.path.basename(file_path))[0]
            plot(markers, "markers_" + title)

        plot_markers()

    # @skip("Takes a lot of time to generate graphs")
    def test_plot_segment_labels(self):
        @iter_all_files(__dir__ + '/data/labels')
        def plot_labels(file_path: str):
            labels = np.load(file_path)
            title = os.path.splitext(os.path.basename(file_path))[0]
            plot(labels, "labels_" + title)

        plot_labels()


