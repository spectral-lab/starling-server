from PIL import Image
import numpy as np
from skimage import img_as_float
import io
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from .modules import segmentize, detect_peaks, format_check

print('Successfully imported')

app = Flask(__name__)
CORS(app)


@app.route('/', methods=["POST"])
def handler():
    print('Success: Got request')
    uploaded_img = request.data
    image_data = Image.open(io.BytesIO(uploaded_img)).convert('L')
    img = img_as_float(image_data) / 255.
    check_result = format_check(img)
    if not check_result.is_ok:
        print(check_result)
        return check_result['msg']
    print('Success: Converted into ndarray')
    labels = segmentize(img)
    print('Success: Segmentized with ' + (labels.max()+1) + " segments")
    peak_points = detect_peaks(img, labels)
    print('Success: Detected ' + peak_points.size[0] + 'peaks')
    ret = make_response(convert_to_json(peak_points))
    print('Success: Returns' + ret)
    return ret


def convert_to_json(in_list):
    print("converting to json")
    return jsonify(list)
