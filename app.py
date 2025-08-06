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

def extract_grains(image):
    """ تقطيع حبوب الأرز من الصورة باستخدام OpenCV (يفترض خلفية سوداء أو بيضاء) """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grains = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # تجاهل الأجسام الصغيرة جداً (ليست حبوب أرز)
        if w < 10 or h < 10:
            continue
        grain_img = image[y:y+h, x:x+w]
        grains.append(grain_img)
    return grains

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    rice_grains = extract_grains(img)
    if len(rice_grains) == 0:
        return jsonify({'error': 'No grains detected'}), 400

    pred_counts = {name: 0 for name in classes}
    for grain_img in rice_grains:
        try:
            g_img = cv2.resize(grain_img, (224, 224)) / 255.0
            g_img = np.expand_dims(g_img, axis=0)
            pred = model.predict(g_img, verbose=0)
            pred_class = classes[np.argmax(pred)]
            pred_counts[pred_class] += 1
        except Exception as e:
            continue

    total = sum(pred_counts.values())
    distribution = []
    for name in classes:
        pct = (pred_counts[name] / total * 100) if total else 0
        distribution.append({"variety": name, "count": pred_counts[name], "percentage": round(pct, 1)})

    majority = max(pred_counts, key=pred_counts.get)
    majority_pct = (pred_counts[majority] / total * 100) if total else 0

    # إذا في نوع آخر غير الأغلب وله نسبة > 0
    mixture = [name for name in classes if name != majority and pred_counts[name] > 0]
    mixture_summary = ', '.join(mixture) if mixture else "None"

    result = {
        "total_grains": total,
        "majority_variety": f"{majority} ({majority_pct:.1f}%)",
        "mixture_variety": mixture_summary,
        "distribution": distribution
    }
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
