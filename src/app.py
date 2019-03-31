#pylint: disable=import-error
from PIL import Image
import numpy as np
import io
import librosa
from skimage.segmentation import random_walker
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.measure import label
from skimage import img_as_float
from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=["POST"])
def handler():
    print('got request')
    uploaded_img = request.data
    image_data = Image.open(io.BytesIO(uploaded_img))
    labels = segmentize(image_data)
    print(labels)
    partials = detect_partial(image_data, labels)
    ret = make_response(convert_to_json(labels))
    print(ret)
    return ret

def segmentize(image_data):
    im = np.array(image_data)
    data = img_as_float(im)
    normalized_im = rescale_intensity(im, in_range=(0, 1), out_range=(-1, 1))
    smoothed_im = gaussian(normalized_im, sigma=(8, 0.4))
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[smoothed_im < -0.5] = 1
    markers[smoothed_im > -0.1] = 2
    binary_segment = random_walker(smoothed_im, markers, beta=10, mode='bf')
    labels = label(np.array(binary_segment)) - 2
    print('got labels')
    return labels

def detect_partials(S, labels):
    times = librosa.frames_to_time(np.arange(S.shape[1]))
    freqs = librosa.fft_frequencies()
    time_grid, freq_grid = np.meshgrid(times, freqs)
    partials = []
    for i in range(labels.max()):
        segment = labels == i
        partial_positions = find_partial_in_segment(S, segment)
        partial = [dict(time=time_grid[position],
                        freq=freq_grid[position],
                        amp=S[position])
                             for position in partial_positions]
        partials.append(partial)


def find_partial_in_segment(S, segment):
    i, j = np.where(segment)
    segment_columns = np.unique(j)
    partial_positions = []
    for column_idx in segment_columns:
        amp_column = S[:, column_idx]
        is_segment = segment[:, column_idx]
        max_val = amp_column[is_segment].max()
        row_idx = np.where(amp_column == max_val)[0][0]
        partial_positions.append((row_idx, column_idx))
    return partial_positions


def convert_to_json(np_array):
    print("converting to json")
    np_list = np_array.tolist()
    return jsonify(segments=np_list)