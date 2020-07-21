import cv2
import os

base_path = '/home/nebula/juneli/pytorch-project/yolov5/inference/videos/getvideo_189_2019-07-20_09-17-48.avi'
cap = cv2.VideoCapture(base_path)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('image', frame)
    cv2.waitKey(1)
    cv2.imwrite('/home/nebula/juneli/pytorch-project/yolov5/inference/videos-frame/' + str(count).zfill(6) + '.jpg', frame)
    count += 1
