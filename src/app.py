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
# noinspection PyPackageRequirements,PyUnresolvedReferences
from modules import segmentize, check_format, detect_peaks, compute_marks, export_graph, format_as_2d_array
import pytz

print('Success: Imported modules')

app = Flask(__name__)
CORS(app)
line_continuity = 1  # This will be taken from the request from client


@app.route('/', methods=["POST"])
def handler():
    def export_intermediate_data():
        peaks_2d = format_as_2d_array(peak_points, spectrogram.shape)
        now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        export_graph(spectrogram,
                     "spectrogram_" + now.strftime("%Y%m%d_%H%M"))  # ex: spectrogram_20190417_0910
        export_graph(markers, "markers_" + now.strftime("%Y%m%d_%H%M"))
        export_graph(labels, "labels_" + now.strftime("%Y%m%d_%H%M"))
        export_graph(peaks_2d, "peak_points_" + now.strftime("%Y%m%d_%H%M"))
        print("Success: Exported graphs")

    print('Success: Got request')
    uploaded_img = request.data
    image_data = Image.open(io.BytesIO(uploaded_img)).convert('L')
    spectrogram = img_as_float(image_data) / 255.
    check_result = check_format(spectrogram)
    if not check_result["is_ok"]:
        print("bad format")
        print(check_result['msg'])
        return check_result['msg']
    print('Success: Converted into ndarray')
    markers = compute_marks(spectrogram)
    labels = segmentize(spectrogram, markers, line_continuity)
    print('Success: Segmentized with ' + str(labels.max() + 1) + " segments to extract peaks")
    peak_points = detect_peaks(spectrogram, labels)
    print('Success: Detected ' + str(len(peak_points)) + ' peaks')
    ret = make_response(jsonify(peak_points))

    export_intermediate_data()  # Optional

    print('Success: Returns')
    print(str(peak_points))
    return ret

