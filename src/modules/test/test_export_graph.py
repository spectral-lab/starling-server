from unittest import TestCase, skip
from .. import export_graph, format_as_2d_array
import numpy as np
import os
from .helpers import iter_all_files
from pdb import set_trace

__dirname__ = os.path.dirname(os.path.realpath(__file__))


# noinspection PyTypeChecker
class TestExportGraph(TestCase):
    # @skip("Takes a lot of time to generate graphs")
    def test_export_graph_spectrogram(self):
        @iter_all_files(__dirname__ + '/data/spectrogram')
        def export_spectrogram(file_path):
            spectrogram_img = np.load(file_path)
            title = os.path.splitext(os.path.basename(file_path))[0]
            export_graph(spectrogram_img, "spectrogram_" + title)

        export_spectrogram()

    # @skip("Takes a lot of time to generate graphs")
    def test_export_marks(self):
        @iter_all_files(__dirname__ + '/data/seed_markers')
        def export_seed_markers(file_path):
            seed_markers = np.load(file_path)
            title = os.path.splitext(os.path.basename(file_path))[0]
            export_graph(seed_markers, "seed_markers_" + title)

        export_seed_markers()

    # @skip("Takes a lot of time to generate graphs")
    def test_export_segment_labels(self):
        @iter_all_files(__dirname__ + '/data/labels')
        def export_labels(file_path: str):
            labels = np.load(file_path)
            title = os.path.splitext(os.path.basename(file_path))[0]
            export_graph(labels, "labels_" + title)

        export_labels()

    def test_export_peak_points(self):
        @iter_all_files(__dirname__ + '/data/peak_points')
        def export_peak_points(file_path: str):
            peak_points = np.load(file_path).tolist()
            spectrogram_img = np.load(os.path.join(__dirname__, 'data/spectrogram', os.path.basename(file_path)))
            peak_indices = format_as_2d_array(peak_points, spectrogram_img.shape)
            title = os.path.splitext(os.path.basename(file_path))[0]
            export_graph(peak_indices, "peak_points_" + title)

        export_peak_points()


