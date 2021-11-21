from flask import Flask,send_file, request, jsonify
from yolo_detection_images import mainDetector 
from PIL import Image
app = Flask(__name__)

@app.route("/detect", methods=["POST"])
def process_image():
    userId=request.headers["userId"]
    file = request.files['image']
    # Read the image via file.stream
    file.save('images/im-received.jpg')
    res= mainDetector('images/im-received.jpg',userId)
    #return jsonify({'Quantity': res})
    return send_file("images/test"+userId+".jpg",as_attachment=True,attachment_filename='test.jpg',mimetype='image/jpg')
    #return res

if __name__ == "__main__":
    app.run(threaded=True, port=5000)
