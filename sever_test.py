from flask import Flask, request, jsonify
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from flask_cors import CORS
# from APIKeyMiddleware import APIKeyMiddleware
from dotenv import load_dotenv
import os
from werkzeug.wrappers import Response

class APIKeyMiddleware:
    def __init__(self, app, valid_api_key):
        self.app = app
        self.valid_api_key = valid_api_key

    def __call__(self, environ, start_response):
        # Check if the request method is OPTIONS
        if environ.get('REQUEST_METHOD') == 'OPTIONS':
            # Return a CORS-compliant response for preflight requests
            response = Response('', status=200)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'X-API-KEY, Content-Type'
            return response(environ, start_response)

        # Existing API key validation for non-OPTIONS requests
        api_key = environ.get('HTTP_X_API_KEY')
        if not api_key or api_key != self.valid_api_key:
            response = Response('Invalid or missing API key', status=403)
            return response(environ, start_response)

        return self.app(environ, start_response)

load_dotenv()
KEY = os.getenv("API_KEY")
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})
# Định nghĩa khóa API hợp lệ
VALID_API_KEY = KEY

# Áp dụng middleware vào ứng dụng Flask
app.wsgi_app = APIKeyMiddleware(app.wsgi_app, VALID_API_KEY)

# Load mô hình
model = load_model("last.h5")

# Danh sách nhãn lớp
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image):
    """Tiền xử lý ảnh để đưa vào model"""
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    """Nhận nhiều ảnh từ request, dự đoán bằng mô hình, và trả về kết quả"""
    if 'images' not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    files = request.files.getlist('images')
    results = []

    # Dự đoán cho từng ảnh
    for file in files:
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))

            results.append({
                "diagnosis": class_labels[predicted_class],
                "confidence": confidence
            })
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    # Tổng hợp kết quả
    disease_count = {'glioma': 0, 'meningioma': 0, 'notumor': 0, 'pituitary': 0}
    significant_diagnoses = []

    # Đếm số lượng từng loại chẩn đoán
    for result in results:
        if result['confidence'] > 0.5:  # Ngưỡng confidence
            disease_count[result['diagnosis']] += 1
            if result['diagnosis'] != 'notumor':
                if(result['diagnosis'] not in [d['disease'] for d in significant_diagnoses]):
                    significant_diagnoses.append({
                        "disease": result['diagnosis'],
                        "confidence": result['confidence']
                    })

    # Tạo summary
    if len(significant_diagnoses) == 0:
        summary = {
            "text": "Kết quả dự đoán",
            "total": "Bình thường",
            "details": "Không phát hiện bệnh nào khác thường"
        }
    else:
        detected_diseases = [d['disease'] for d in significant_diagnoses]
        summary = {
            "text": "Kết quả dự đoán",
            "total": f"Phát hiện {len(significant_diagnoses)} vấn đề",
            "details": {
                "disease_counts": disease_count,
                "detected_diseases": detected_diseases
            }
        }

    response_data = {"results": results, "summary": summary}
    print("Response:", response_data)  # In dữ liệu để debug
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)