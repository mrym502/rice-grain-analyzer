from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# إذا الملف غير موجود، حمّليه من Google Drive
if not os.path.exists('rice_resnet_model.h5'):
    import gdown
    # استخدمي الرابط المباشر كما يأتي:
    url = "https://drive.google.com/uc?id=1EssBqvWlPsPZuAFR6c68jYM9g6qe0f87"
    gdown.download(url, 'rice_resnet_model.h5', quiet=False)

# تحميل النموذج
model = tf.keras.models.load_model('rice_resnet_model.h5')

classes = [
    "1121", "1401", "1509", "PR-11", "RH-10",
    "Sharbati", "Sona Masoori", "Sugandha"
]

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    var = classes[np.argmax(predictions)]
    conf = float(np.max(predictions)) * 100

    return jsonify({'variety': var, 'confidence': conf})

if __name__ == '__main__':
    app.run(debug=True)
