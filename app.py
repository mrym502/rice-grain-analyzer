from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import requests

# تحميل النموذج من Dropbox إذا لم يكن موجوداً
def download_model():
    url = "https://www.dropbox.com/scl/fi/53x6hzdnv5nleagwafhqw/rice_resnet_model.h5?rlkey=vtkxm7cf867d3lhodb4abzl9i&st=jw2dfskz&dl=1"
    response = requests.get(url)
    with open('rice_resnet_model.h5', 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully.")

if not os.path.exists('rice_resnet_model.h5'):
    print("Model file not found, downloading from Dropbox...")
    download_model()
else:
    print("Model file already exists.")

app = Flask(__name__)

# تحميل النموذج المدرب
model = tf.keras.models.load_model('rice_resnet_model.h5')
classes = [
    "1121",
    "1401",
    "1509",
    "PR-11",
    "RH-10",
    "Sharbati",
    "Sona Masoori",
    "Sugandha"
]

@app.route('/analyze', methods=['POST'])
def analyze():
    print("FILES:", request.files)
    print("FORM:", request.form)
    print("CONTENT_TYPE:", request.content_type)
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    print("BYTES LEN:", len(img_bytes))
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = classes[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    result = {
        'variety': predicted_class,
        'confidence': confidence
    }
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
