import datetime
import requests
import jwt
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify

load_dotenv()
API_KEY = os.getenv("API_KEY")
TOKEN_KEY = os.getenv("TOKEN_KEY")
app = Flask(__name__)
app.config['SECRET_KEY'] = TOKEN_KEY
INTERNAL_API = 'http://127.0.0.1:5000/predict'


# Cho phép CORS nếu frontend ở domain khác
from flask_cors import CORS
CORS(app)

# Route login → tạo token
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    if data['username'] == 'admin' and data['password'] == '123456':
        token = jwt.encode({
            'user': data['username'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        return jsonify({'token': token})
    return jsonify({'error': 'Sai thông tin đăng nhập'}), 401

# Route gọi /predict (có kiểm tra token)
@app.route('/api/predict', methods=['POST'])
def proxy_predict():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'Thiếu token'}), 401
    try:
        token = auth_header.split(" ")[1]
        jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
    except Exception as e:
        return jsonify({'error': 'Token không hợp lệ'}), 401

    files = request.files.getlist('images')
    if len(files) == 0:
        return jsonify({'error': 'Không có ảnh nào được tải lên'}), 400

    files_to_send = [('images', (file.filename, file.stream, file.content_type)) for file in files]
    headers = {'X-API-KEY': API_KEY}
    response = requests.post(INTERNAL_API, files=files_to_send, headers=headers)
    if response.status_code != 200:
        return jsonify({'error': 'Lỗi từ dịch vụ nội bộ', 'details': response.text}), response.status_code
    return jsonify(response.json()), response.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000, debug=True)
