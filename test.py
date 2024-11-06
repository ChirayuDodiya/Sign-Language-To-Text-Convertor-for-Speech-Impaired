import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 224
offset = 30
margin = 20

# Load two separate models for left and right hands
classifier_left = Classifier("Model/keras_model_left.h5", "Model/labels_left.txt")
classifier_right = Classifier("Model/keras_model_right.h5", "Model/labels_right.txt")

fingerColors = {
    'thumb': (0, 0, 255), 
    'index': (0, 255, 0), 
    'middle': (255, 0, 0),
    'ring': (0, 250, 250),
    'pinky': (255, 0, 255)
}

nodeColors = [
    (0, 0, 200),
    (0, 0, 170),
    (0, 0, 140),
    (0, 0, 110),

    (0, 0, 80),

    (0, 200, 0),
    (0, 170, 0),
    (0, 140, 0),
    (0, 110, 0),

    (200, 0, 0),
    (170, 0, 0),
    (140, 0, 0),
    (110, 0, 0),

    (0, 200, 200),
    (0, 170, 170),
    (0, 140, 140),
    (0, 110, 110),

    (200, 0, 200),
    (170, 0, 170),
    (140, 0, 140),
    (110, 0, 110),
]

orange = (0, 165, 255)
black = (0, 0, 0)

fingerConnections = {
    'thumb': [(1, 2), (2, 3), (3, 4)],
    'index': [(5, 6), (6, 7), (7, 8)],
    'middle': [(9, 10), (10, 11), (11, 12)],
    'ring': [(13, 14), (14, 15), (15, 16)],
    'pinky': [(17, 18), (18, 19), (19, 20)]
}

wristIndex = 0
fingertipIndexes = [1, 5, 9, 13, 17]

def crop_resize_img(lmList, imgSize, margin, handType):
    xList = [lm[0] for lm in lmList]
    yList = [lm[1] for lm in lmList]
    xMin, xMax = min(xList) - margin, max(xList) + margin
    yMin, yMax = min(yList) - margin, max(yList) + margin

    width, height = xMax - xMin, yMax - yMin
    aspectRatio = height / width

    try:
        if aspectRatio > 1:
            k = imgSize / height
            wCal = math.ceil(k * width)
            imgResize = np.ones((imgSize, wCal, 3), np.uint8) * 255
            wGap = math.ceil((imgSize - wCal) / 2)

            for finger, connections in fingerConnections.items():
                color = fingerColors[finger]
                for connection in connections:
                    point1 = lmList[connection[0]]
                    point2 = lmList[connection[1]]

                    x1 = np.interp(point1[0], [xMin, xMax], [0, wCal])
                    y1 = np.interp(point1[1], [yMin, yMax], [0, imgSize])

                    x2 = np.interp(point2[0], [xMin, xMax], [0, wCal])
                    y2 = np.interp(point2[1], [yMin, yMax], [0, imgSize])

                    cv2.line(imgResize, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            wristColor = orange if handType == 'Left' else black
            for fingertip in fingertipIndexes:
                point1 = lmList[wristIndex]
                point2 = lmList[fingertip]

                x1 = np.interp(point1[0], [xMin, xMax], [0, wCal])
                y1 = np.interp(point1[1], [yMin, yMax], [0, imgSize])

                x2 = np.interp(point2[0], [xMin, xMax], [0, wCal])
                y2 = np.interp(point2[1], [yMin, yMax], [0, imgSize])

                cv2.line(imgResize, (int(x1), int(y1)), (int(x2), int(y2)), wristColor, 2)

            for i in range(len(fingertipIndexes) - 1):
                point1 = lmList[fingertipIndexes[i]]
                point2 = lmList[fingertipIndexes[i + 1]]

                x1 = np.interp(point1[0], [xMin, xMax], [0, wCal])
                y1 = np.interp(point1[1], [yMin, yMax], [0, imgSize])

                x2 = np.interp(point2[0], [xMin, xMax], [0, wCal])
                y2 = np.interp(point2[1], [yMin, yMax], [0, imgSize])

                cv2.line(imgResize, (int(x1), int(y1)), (int (x2), int(y2)), wristColor, 2)

            for i, point in enumerate(lmList):
                x = np.interp(point[0], [xMin, xMax], [0, wCal])
                y = np.interp(point[1], [yMin, yMax], [0, imgSize])
                cv2.circle(imgResize, (int(x), int(y)), 5, nodeColors[i], cv2.FILLED)

            FinalImage = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            FinalImage[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / width
            hCal = math.ceil(k * height)
            imgResize = np.ones((hCal, imgSize, 3), np.uint8) * 255
            hGap = math.ceil((imgSize - hCal) / 2)

            for finger, connections in fingerConnections.items():
                color = fingerColors[finger]
                for connection in connections:
                    point1 = lmList[connection[0]]
                    point2 = lmList[connection[1]]

                    x1 = np.interp(point1[0], [xMin, xMax], [0, imgSize])
                    y1 = np.interp(point1[1], [yMin, yMax], [0, hCal])

                    x2 = np.interp(point2[0], [xMin, xMax], [0, imgSize])
                    y2 = np.interp(point2[1], [yMin, yMax], [0, hCal])

                    cv2.line(imgResize, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            wristColor = orange if handType == 'Left' else black
            for fingertip in fingertipIndexes:
                point1 = lmList[wristIndex]
                point2 = lmList[fingertip]

                x1 = np.interp(point1[0], [xMin, xMax], [0, imgSize])
                y1 = np.interp(point1[1], [yMin, yMax], [0, hCal])

                x2 = np.interp(point2[0], [xMin, xMax], [0, imgSize])
                y2 = np.interp(point2[1], [yMin, yMax], [0, hCal])

                cv2.line(imgResize, (int(x1), int(y1)), (int(x2), int(y2)), wristColor, 2)

            for i in range(len(fingertipIndexes) - 1):
                point1 = lmList[fingertipIndexes[i]]
                point2 = lmList[fingertipIndexes[i + 1]]

                x1 = np.interp(point1[0], [xMin, xMax], [0, imgSize])
                y1 = np.interp(point1[1], [yMin, yMax], [0, hCal])

                x2 = np.interp(point2[0], [xMin, xMax], [0, imgSize])
                y2 = np.interp(point2[1], [yMin, yMax], [0, hCal])

                cv2.line(imgResize, (int(x1), int(y1)), (int(x2), int(y2)), wristColor, 2)

            for i, point in enumerate(lmList):
                x = np.interp(point[0], [xMin, xMax], [0, imgSize])
                y = np.interp(point[1], [yMin, yMax], [0, hCal])
                cv2.circle(imgResize, (int(x), int(y)), 5, nodeColors[i], cv2.FILLED)

            FinalImage = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            FinalImage[hGap:hCal + hGap, :] = imgResize

    except ValueError as e:
        print(f"Error during image resizing: {e}")
        FinalImage = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    return FinalImage

with open("Model/labels_left.txt", "r") as f:
    labels_dict_left = {}
    for line in f.readlines():
        idx, label = line.strip().split()
        labels_dict_left[int(idx)] = label

with open("Model/labels_right.txt", "r") as f:
    labels_dict_right = {}
    for line in f.readlines():
        idx, label = line.strip().split()
        labels_dict_right[int(idx)] = label

output_string = ""
font_size = 1
font_thickness = 2
max_string_length = 28
frame_count = 11

consecutive_predictions = []
consecutive_count = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        handType = hand["type"]

        FinalImage = crop_resize_img(lmList, imgSize, margin, handType)

        if handType == "Left":
            _, prediction = classifier_left.getPrediction(FinalImage)
        else:
            _, prediction = classifier_right.getPrediction(FinalImage)

        if consecutive_predictions and prediction == consecutive_predictions[-1]:
            consecutive_count += 1
        else:
            consecutive_predictions = [prediction]
            consecutive_count = 1

        if consecutive_count == frame_count:
            if handType == "Left":
                label = labels_dict_left[prediction]
            else:
                label = labels_dict_right[prediction]
            if label == "del":
                if output_string:
                    output_string = output_string[:-1]
            else:
                output_string += label
            consecutive_predictions = []
            consecutive_count = 0

    if len(output_string) > max_string_length:
        output_string = output_string[-max_string_length:]

    cv2.putText(img, output_string, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness )

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
