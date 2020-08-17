import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import easygui
import os
import six
easygui.buttonbox('识别需要一段时间，请耐心等待，点击图片开始识别',image="./image/welcome.gif",choices=())
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
confidenceThreshold = 0.5
NMSThreshold = 0.3
modelConfiguration = './yolov3/yolov3.cfg'
modelWeights = './yolov3/yolov3.weights'
#　获取标签
labelsPath = './yolov3/coco.names'
labels = open(labelsPath).read().strip().split('\n')
# 产生随机颜色
np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
# 读取DNN网络
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# 获取输出层
outputLayer = net.getLayerNames()
outputLayer = [outputLayer[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# 加载视频
video = cv2.VideoCapture(file_path)
writer = None
(W, H) = (None, None)
try:
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(video.get(prop))
    print("[INFO] {} 为视频帧数 ".format(total))
    print('预计所需时间约:'+str(total)+'秒')
except:
    print("无法确定视频中的帧数")
count = 0
while True:
    (ret, frame) = video.read()
    if not ret:
        break
    if W is None or H is None:
        (H,W) = frame.shape[:2]
    # DNN数据转换
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(outputLayer)
    boxes = []
    confidences = []
    classIDs = []
    # 输出层
    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # NMS算法获取目标
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
    if(len(detectionNMS) > 0):
        # 目标平化
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if writer is None:
                # 视频格式
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter('./result/output.avi', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    if writer is not None:
        writer.write(frame)
        print("车辆识别中")
        print("Writing frame", count+1)
        count = count + 1
        if count == total:
            print('识别结束')
            os.system('python TF.py')