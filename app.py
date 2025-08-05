from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

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
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
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
    app.run(debug=True)
