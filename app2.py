from flask import Flask, request, jsonify
import cv2
import os
import easyocr
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Load your trained YOLO models for sticker and serial number detection
sticker_model = YOLO('./models/sticker_detection_model.pt')
serial_number_model = YOLO('./models/yolosno.pt')

# Function to get the rotation angle using Hough Line Transform
def get_rotation_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        rho, theta = lines[0][0]
        angle = np.degrees(theta) - 90
        return angle
    return 0

# Function to rotate the image based on the detected angle
def rotate_image(image, angle):
    if angle == 0:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated_image

# Function to detect serial number region
def detect_serial_number(cropped_rotated_image):
    results = serial_number_model.predict(cropped_rotated_image)
    boxes = results[0].boxes
    if len(boxes) > 0:
        box = boxes.xyxy[0]
        x1, y1, x2, y2 = box
        serial_number_crop = cropped_rotated_image[int(y1):int(y2), int(x1):int(x2)]
        return serial_number_crop
    return None

# Function to extract serial number using EasyOCR
def extract_serial_number(image):
    results = reader.readtext(image, detail=0)
    for text in results:
        if text.isdigit() and 4 <= len(text) <= 10:
            return text
    return "not detected"

# Function to process the image using YOLO and then apply rotation to the cropped regions
def process_image(image):
    results = sticker_model.predict(image)
    boxes = results[0].boxes
    scores = boxes.conf
    classes = boxes.cls

    for box, score, cls in zip(boxes.xyxy, scores, classes):
        x1, y1, x2, y2 = box
        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
        angle = get_rotation_angle(cropped_image)
        rotated_cropped_image = rotate_image(cropped_image, angle)
        serial_number_image = detect_serial_number(rotated_cropped_image)

        if serial_number_image is not None:
            detected_serial_number = extract_serial_number(serial_number_image)
            return detected_serial_number

    return "not detected"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image to a temporary location
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Load the image
    image = cv2.imread(file_path)

    # Process the image
    detected_serial_number = process_image(image)

    return jsonify({"serial_number": detected_serial_number})

if __name__ == '__main__':
    app.run(debug=True)
