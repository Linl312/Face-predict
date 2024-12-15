#python app.py

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2 as cv
import numpy as np
import time
import os
from predictions import get_face_prediction  # 这里将导入预测函数
print("Checking if files exist:")

app = Flask(__name__)
socketio = SocketIO(app)

# 设置文件上传路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 人脸识别与预测模型
faceProto = "source/opencv_face_detector.pbtxt"
faceModel = "source/opencv_face_detector_uint8.pb"

ageProto = "source/age_deploy.prototxt"
ageModel = "source/age_net.caffemodel"

genderProto = "source/gender_deploy.prototxt"
genderModel = "source/gender_net.caffemodel"


print("Checking if files exist:")
print(os.path.exists(faceProto))  # True 如果文件存在，False 如果不存在
print(os.path.exists(faceModel))
print(os.path.exists(ageProto))
print(os.path.exists(ageModel))
print(os.path.exists(genderProto))
print(os.path.exists(genderModel))


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# 加载网络模型
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# 路由: 首页
@app.route('/')
def index():
    return render_template('index.html')

# 路由: 处理图像上传
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # 进行人脸年龄性别预测
        prediction = get_face_prediction(filename)

        return jsonify({'prediction': prediction})

# WebSocket 监听: 接收前端图像
@socketio.on('image_data')
def handle_image_data(image_data):
    # 将图像数据处理并进行预测
    prediction = get_face_prediction(image_data)
    emit('prediction_result', prediction)

if __name__ == '__main__':
    socketio.run(app, debug=True)
