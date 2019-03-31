#pylint: disable=import-error
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage
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
    ret = make_response(convert_to_json(labels))
    print(ret)
    return ret

def segmentize(image_data):
    im = np.array(image_data)
    data = skimage.img_as_float(im)
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < -0.95] = 1
    markers[data > 0.95] = 2
    labels = random_walker(data, markers, beta=10, mode='bf')
    print('got labels')
    return labels

def convert_to_json(np_array):
    print("converting to json")
    np_list = np_array.tolist()
    return jsonify(segments=np_list)