from unittest import TestCase, skip
from .. import segmentize, plot
from skimage.data import binary_blobs
import numpy as np
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from skimage.measure import label
import os
import random
import librosa

TARGET_ACCURACY_RATE = 0.85
NUMBER_OF_TEST_LOOPS = 30


# noinspection PyTypeChecker
class TestSegmentize(TestCase):
    # @skip("")
    def test_accuracy_rate(self):
        accuracy_rate = 0
        for _ in range(NUMBER_OF_TEST_LOOPS):
            specimen = img_as_float(binary_blobs(length=16, seed=1))
            expected_answer = label(specimen).max()
            sigma = 0.35
            specimen += 0.07 * np.random.normal(loc=0, scale=sigma, size=specimen.shape)
            specimen = rescale_intensity(specimen, in_range=(specimen.min(), specimen.max()), out_range=(0., 1.))
            line_continuity = 0
            peak_level = 0.9
            background_level = 0.1
            labels = segmentize(specimen, line_continuity, peak_level, background_level)
            number_of_segments = labels.max() + 1
            if number_of_segments == expected_answer:
                accuracy_rate += 1. / NUMBER_OF_TEST_LOOPS
        if accuracy_rate < TARGET_ACCURACY_RATE:
            print('accuracy rate is')
            print(accuracy_rate)
        self.assertTrue(accuracy_rate > TARGET_ACCURACY_RATE)

    # @skip("")
    def test_smoothing(self):
        specimen = img_as_float(binary_blobs(length=16, seed=1))
        sigma = 0.35
        specimen += 0.07 * np.random.normal(loc=0, scale=sigma, size=specimen.shape)
        specimen = rescale_intensity(specimen, in_range=(specimen.min(), specimen.max()), out_range=(0., 1.))
        line_continuity = 2
        peak_level = 0.85
        background_level = 0.1
        labels = segmentize(specimen, line_continuity, peak_level, background_level)
        number_of_segments = labels.max() + 1
        self.assertTrue(number_of_segments > 0)

    # @skip("")
    def test_real_data(self):
        filename = random.choice(os.listdir("./src/modules/test/data/img_from_client"))
        spectrogram_img = np.load('./src/modules/test/data/img_from_client/' + filename)
        line_continuity = 2
        peak_level = 0.85
        background_level = 0.1
        labels = segmentize(spectrogram_img, line_continuity, peak_level, background_level)
        number_of_segments = labels.max() + 1
        self.assertTrue(number_of_segments > 0)

    @skip("Takes a lot of time to generate graphs")
    def test_plot(self):
        """
        This test is aimed at generating graphs of spectrogram and segment labels into `./output` folder.
        Taking a lot of time, please use it only when you need to see the graphs.
        """
        filename = random.choice(os.listdir("./src/modules/test/data/img_from_client"))
        spectrogram_img = np.load('./src/modules/test/data/img_from_client/' + filename)

        plot(spectrogram_img, "spectrogram_" + filename.replace(".npy", ""))

        line_continuity = 2
        peak_level = 0.85
        background_level = 0.1
        labels = segmentize(spectrogram_img, line_continuity, peak_level, background_level)

        plot(labels, "labels_" + filename.replace(".npy", ""))

        number_of_segments = labels.max() + 1
        self.assertTrue(number_of_segments > 0)
