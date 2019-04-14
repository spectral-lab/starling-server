from unittest import TestCase, main
from .. import segmentize
from skimage.data import binary_blobs
import numpy as np
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from skimage.measure import label

TARGET_ACCURACY_RATE = 0.85
NUMBER_OF_LOOP = 100


# noinspection PyTypeChecker
class TestSegmentize(TestCase):
    def test_accuracy_rate(self):
        accuracy_rate = 0
        for _ in range(NUMBER_OF_LOOP):
            specimen = img_as_float(binary_blobs(length=16, seed=1))
            expected_answer = label(specimen).max()
            sigma = 0.35
            specimen += 0.07 * np.random.normal(loc=0, scale=sigma, size=specimen.shape)
            specimen = rescale_intensity(specimen, in_range=(specimen.min(), specimen.max()), out_range=(0., 1.))
            line_continuity = 0
            peak_level = 0.9
            background_level = 0.1
            # np.save(('src/modules/test/data/spectrogram2d/' + str(_) + '.npy'), specimen)
            labels = segmentize(specimen, line_continuity, peak_level, background_level)
            # np.save(('src/modules/test/data/labels/' + str(_) + '.npy'), labels)
            number_of_segments = labels.max() + 1
            if number_of_segments == expected_answer:
                accuracy_rate += 1. / NUMBER_OF_LOOP
        if accuracy_rate < TARGET_ACCURACY_RATE:
            print('accuracy rate is')
            print(accuracy_rate)
        self.assertTrue(accuracy_rate > TARGET_ACCURACY_RATE)

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
