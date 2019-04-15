import sys
import os

__dirname = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(__dirname, "/modules")))

PACKAGE_PARENT = '..'
from PIL import Image
from skimage import img_as_float
import io
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from modules import segmentize, format_check, detect_peaks
from librosa.core import db_to_power
import numpy as np
print('Successfully imported')

app = Flask(__name__)
CORS(app)


@app.route('/', methods=["POST"])
def handler():
    line_continuity = 1
    peak_level = db_to_power(-30)
    background_level = db_to_power(-35)
    print(peak_level, background_level)
    print('Success: Got request')
    uploaded_img = request.data
    image_data = Image.open(io.BytesIO(uploaded_img)).convert('L')
    img = img_as_float(image_data) / 255.
    # np.save('./src/modules/test/data/img_from_client/era.npy', img)
    check_result = format_check(img)
    if not check_result["is_ok"]:
        print("bad format")
        print(check_result)
        return check_result['msg']
    print('Success: Converted into ndarray')
    labels = segmentize(img, line_continuity, peak_level, background_level)
    print('Success: Segmentized with ' + str(labels.max() + 1) + " segments to extract peaks")
    peak_points = detect_peaks(img, labels)
    print('Success: Detected ' + str(len(peak_points)) + ' peaks')
    ret = make_response(convert_to_json(peak_points))
    print('Success: Returns ' + str(peak_points))
    return ret


def convert_to_json(in_list):
    return jsonify(in_list)
