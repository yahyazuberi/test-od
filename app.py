from flask import Flask, request, jsonify
from yolo_detection_images import mainDetector 
from PIL import Image
app = Flask(__name__)

@app.route("/im_size", methods=["POST"])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    file.save('images/im-received.jpg')
    res=mainDetector('images/im-received.jpg')
    return jsonify({'Quantity': res})


if __name__ == "__main__":
    app.run(threaded=True, port=5000)
