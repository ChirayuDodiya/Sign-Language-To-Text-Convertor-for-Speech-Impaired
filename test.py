import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 30
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, width, height = hand['bbox']
        FinalImage = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + height + offset, x - offset:x + width + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = height / width

        if aspectRatio > 1:
            k = imgSize / height
            wCal = math.ceil(k * width)
            try:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                FinalImage[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(FinalImage, draw=False)
                print(prediction, index)
            except cv2.error as e:
                print(f"Error while resizing: {e}")
            except ValueError as ve:
                print(f"ValueError: {ve}")
                
        else:
            k = imgSize / width
            hCal = math.ceil(k * height)
            try:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                FinalImage[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(FinalImage, draw=False)
            except cv2.error as e:
                print(f"Error while resizing: {e}")
            except ValueError as ve:
                print(f"ValueError: {ve}")

        try:
            cv2.rectangle(imgOutput, (x - offset, y - offset-50),(x - offset+150, y - offset), (166, 2, 82), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -35), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        except NameError as ne:
            print(f"NameError: {ne}")
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                    (x + width+offset, y + height+offset), (166, 2, 82), 4)
        
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

