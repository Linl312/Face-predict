import cv2 as cv
import numpy as np

faceProto = "source/opencv_face_detector.pbtxt"
faceModel = "source/opencv_face_detector_uint8.pb"

ageProto = "source/age_deploy.prototxt"
ageModel = "source/age_net.caffemodel"

genderProto = "source/gender_deploy.prototxt"
genderModel = "source/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# 加载网络模型
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

def get_face_prediction(image_path):
    frame = cv.imread(image_path)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # 获取人脸边界框
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])

    predictions = []
    for bbox in bboxes:
        face = frame[max(0, bbox[1]):min(bbox[3], frame.shape[0]), max(0, bbox[0]):min(bbox[2], frame.shape[1])]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # 性别预测
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # 年龄预测
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        predictions.append({'gender': gender, 'age': age})

    return predictions
