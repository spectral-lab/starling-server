import datetime
import pytz
from pdb import set_trace
import numpy as np
from unittest import TestCase
from .. import segmentize, check_format, compute_seeds, export_graph, \
    feature_lines_to_image, compute_feature_lines, extract_training_points
import os

__dirname__ = os.path.dirname(__file__)

line_continuity = 1  # This will be taken from the request from client


class TestOverallProcess(TestCase):
    def test_overall_process(self):
        def export_intermediate_data_as_graph():
            now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
            export_graph(spectrogram,
                         "spectrogram_" + now.strftime("%Y%m%d_%H%M"))  # ex: spectrogram_20190417_0910
            export_graph(markers, "markers_" + now.strftime("%Y%m%d_%H%M"))
            export_graph(segment_labels, "segment_labels_" + now.strftime("%Y%m%d_%H%M"))
            export_graph(feature_lines_to_image(feature_lines, spectrogram.shape),
                         "feature_lines_" + now.strftime("%Y%m%d_%H%M"))

        spectrogram = np.load(__dirname__ + '/data/spectrogram/bird.npy')
        check_result = check_format(spectrogram)
        if not check_result["is_ok"]:
            print("bad format")
            print(check_result['msg'])
            return check_result['msg']
        markers = compute_seeds(spectrogram)
        segment_labels = segmentize(spectrogram, markers, line_continuity)
        training_points = extract_training_points(segment_labels, spectrogram, ratio=0.3)
        feature_lines = compute_feature_lines(training_points, degree=6)

        export_intermediate_data_as_graph()  # Optional
