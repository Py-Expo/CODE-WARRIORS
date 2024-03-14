from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from pytesseract import pytesseract

app = Flask(__name__)

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def recognize_text(image):
    text = pytesseract.image_to_string(Image.fromarray(image), config='--psm 6')
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_text', methods=['POST'])
def capture_text():
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            image_stream = BytesIO(file.read())
            image_stream.seek(0)
            img = np.array(Image.open(image_stream))
            processed_image = preprocess_image(img)
            extracted_text = recognize_text(processed_image)
            return jsonify({'text': extracted_text})
    
    if 'image' in request.json:
        img_data_url = request.json['image']
        img_data = base64.b64decode(img_data_url.split(',')[1])
        img_array = np.array(Image.open(BytesIO(img_data)))
        processed_image = preprocess_image(img_array)
        extracted_text = recognize_text(processed_image)
        return jsonify({'text': extracted_text})

    return jsonify({'error': 'No image found'})

if __name__ == '__main__':
    app.run(debug=True)