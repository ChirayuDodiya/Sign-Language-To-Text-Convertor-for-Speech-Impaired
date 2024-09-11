import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
directory='Data/'
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 30
imgSize = 300
while True:
    count = {
             'a': len(os.listdir(directory+"/A")),
             'b': len(os.listdir(directory+"/B")),
             'c': len(os.listdir(directory+"/C")),
             'd': len(os.listdir(directory+"/D")),
             'e': len(os.listdir(directory+"/E")),
             'f': len(os.listdir(directory+"/F")),
             'g': len(os.listdir(directory+"/G")),
             'h': len(os.listdir(directory+"/H")),
             'i': len(os.listdir(directory+"/I")),
             'j': len(os.listdir(directory+"/J")),
             'k': len(os.listdir(directory+"/K")),
             'l': len(os.listdir(directory+"/L")),
             'm': len(os.listdir(directory+"/M")),
             'n': len(os.listdir(directory+"/N")),
             'o': len(os.listdir(directory+"/O")),
             'p': len(os.listdir(directory+"/P")),
             'q': len(os.listdir(directory+"/Q")),
             'r': len(os.listdir(directory+"/R")),
             's': len(os.listdir(directory+"/S")),
             't': len(os.listdir(directory+"/T")),
             'u': len(os.listdir(directory+"/U")),
             'v': len(os.listdir(directory+"/V")),
             'w': len(os.listdir(directory+"/W")),
             'x': len(os.listdir(directory+"/X")),
             'y': len(os.listdir(directory+"/Y")),
             'z': len(os.listdir(directory+"/Z")),
             }
    success, img = cap.read()
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
            except cv2.error as e:
                print(f"Error while resizing: {e}")
            except ValueError as ve:
                print(f"ValueError: {ve}")

        try:
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("FinalImage", FinalImage)
        except cv2.error as e:
            print(f"Error while resizing: {e}")
    cv2.imshow("Image", img)
    interrupt = cv2.waitKey(5)
