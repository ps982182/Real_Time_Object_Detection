from flask import Flask, Response
import cv2
import numpy as np
from ultralytics import YOLO
import threading

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 nano model

# Global variable to store the video capture
camera = None
lock = threading.Lock()

def generate_frames():
    global camera
    while True:
        with lock:
            if camera is None:
                continue
            success, frame = camera.read()
            if not success:
                continue

            # Perform object detection
            results = model(frame)
            annotated_frame = results[0].plot()  # Draw bounding boxes and labels

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time Object Detection</title>
    </head>
    <body>
        <h1>Real-Time Object Detection</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    """

def start_camera():
    global camera
    with lock:
        if camera is None:
            camera = cv2.VideoCapture(0)  # Use webcam (index 0)

if __name__ == '__main__':
    start_camera()
    app.run(host='0.0.0.0', port=5000, threaded=True)