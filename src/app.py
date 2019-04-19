import sys
import os

__dirname__ = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(__dirname__, "/modules")))

PACKAGE_PARENT = '..'
from PIL import Image
from skimage import img_as_float
import io
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import datetime
import pytz
from pdb import set_trace

# noinspection PyPackageRequirements,PyUnresolvedReferences
from modules import segmentize, check_format, compute_seeds, export_graph, convert_to_json, \
                     format_as_2d_array, compute_feature_lines, extract_training_points

print('Success: Imported modules')

app = Flask(__name__)
CORS(app)
line_continuity = 1  # This will be taken from the request from client


@app.route('/', methods=["POST"])
def handler():
    def export_intermediate_data_as_graph():
        now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        export_graph(spectrogram,
                     "spectrogram_" + now.strftime("%Y%m%d_%H%M"))  # ex: spectrogram_20190417_0910
        export_graph(markers, "markers_" + now.strftime("%Y%m%d_%H%M"))
        export_graph(segment_labels, "segment_labels_" + now.strftime("%Y%m%d_%H%M"))
        print("Success: Exported graphs")

    print('Success: Got request')
    uploaded_img = request.data
    Image.open(io.BytesIO(uploaded_img)).save("./output/img/img_from_client.png")  # Optional. For debugging
    image_data = Image.open(io.BytesIO(uploaded_img)).convert('L')
    spectrogram = img_as_float(image_data) / 255.
    check_result = check_format(spectrogram)
    if not check_result["is_ok"]:
        print("bad format")
        print(check_result['msg'])
        return check_result['msg']
    print('Success: Converted into ndarray')
    markers = compute_seeds(spectrogram)
    segment_labels = segmentize(spectrogram, markers, line_continuity)
    print('Success: Segmentized with ' + str(segment_labels.max() + 1) + " segments to extract peaks")
    training_points = extract_training_points(segment_labels, spectrogram, ratio=0.3)
    feature_lines = compute_feature_lines(training_points, degree=6)
    print('Success: Compute feature lines')
    ret = make_response(convert_to_json(feature_lines))

    export_intermediate_data_as_graph()  # Optional

    print('Success: Returns')
    # print(feature_lines)  # Optional
    return ret
