import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 224
offset = 30
margin = 20  # To prevent point loss at the border

# Load your classifier model
classifier = Classifier("Mod/keras_model.h5", "Mod/labels.txt")

fingerColors = {
    'thumb': (0, 0, 255),   # Red
    'index': (0, 255, 0),   # Green
    'middle': (255, 0, 0),  # Blue
    'ring': (0, 255, 255),  # Yellow
    'pinky': (255, 0, 255)  # Purple
}
orange = (0, 165, 255)  # Orange color for left hand wrist-to-finger connections
black = (0, 0, 0)       # Black color for right hand wrist-to-finger connections

# Connections corresponding to each finger
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

            # Drawing finger connections
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

            # Wrist-to-finger connections
            wristColor = orange if handType == 'Left' else black
            for fingertip in fingertipIndexes:
                point1 = lmList[wristIndex]
                point2 = lmList[fingertip]

                x1 = np.interp(point1[0], [xMin, xMax], [0, wCal])
                y1 = np.interp(point1[1], [yMin, yMax], [0, imgSize])

                x2 = np.interp(point2[0], [xMin, xMax], [0, wCal])
                y2 = np.interp(point2[1], [yMin, yMax], [0, imgSize])

                cv2.line(imgResize, (int(x1), int(y1)), (int(x2), int(y2)), wristColor, 2)

            # Connecting fingertip points (4-8, 8-12, 12-16, 16-20)
            for i in range(len(fingertipIndexes) - 1):
                point1 = lmList[fingertipIndexes[i]]
                point2 = lmList[fingertipIndexes[i + 1]]

                x1 = np.interp(point1[0], [xMin, xMax], [0, wCal])
                y1 = np.interp(point1[1], [yMin, yMax], [0, imgSize])

                x2 = np.interp(point2[0], [xMin, xMax], [0, wCal])
                y2 = np.interp(point2[1], [yMin, yMax], [0, imgSize])

                cv2.line(imgResize, (int(x1), int(y1)), (int(x2), int(y2)), wristColor, 2)

            # Drawing points
            for point in lmList:
                x = np.interp(point[0], [xMin, xMax], [0, wCal])
                y = np.interp(point[1], [yMin, yMax], [0, imgSize])
                cv2.circle(imgResize, (int(x), int(y)), 5, (0, 0, 0), cv2.FILLED)

            FinalImage = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            FinalImage[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / width
            hCal = math.ceil(k * height)
            imgResize = np.ones((hCal, imgSize, 3), np.uint8) * 255
            hGap = math.ceil((imgSize - hCal) / 2)

            # Drawing finger connections
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

            # Wrist-to-finger connections
            wristColor = orange if handType == 'Left' else black
            for fingertip in fingertipIndexes:
                point1 = lmList[wristIndex]
                point2 = lmList[fingertip]

                x1 = np.interp(point1[0], [xMin, xMax], [0, imgSize])
                y1 = np.interp(point1[1], [yMin, yMax], [0, hCal])

                x2 = np.interp(point2[0], [xMin, xMax], [0, imgSize])
                y2 = np.interp(point2[1], [yMin, yMax], [0, hCal])

                cv2.line(imgResize, (int(x1), int(y1)), (int(x2), int(y2)), wristColor, 2)

            # Connecting fingertip points (4-8, 8-12, 12-16, 16-20)
            for i in range(len(fingertipIndexes) - 1):
                point1 = lmList[fingertipIndexes[i]]
                point2 = lmList[fingertipIndexes[i + 1]]

                x1 = np.interp(point1[0], [xMin, xMax], [0, imgSize])
                y1 = np.interp(point1[1], [yMin, yMax], [0, hCal])

                x2 = np.interp(point2[0], [xMin, xMax], [0, imgSize])
                y2 = np.interp(point2[1], [yMin, yMax], [0, hCal])

                cv2.line(imgResize, (int(x1), int(y1)), (int(x2), int(y2)), wristColor, 2)

            # Drawing points
            for point in lmList:
                x = np.interp(point[0], [xMin, xMax], [0, imgSize])
                y = np.interp(point[1], [yMin, yMax], [0, hCal])
                cv2.circle(imgResize, (int(x), int(y)), 5, (0, 0, 0), cv2.FILLED)

            FinalImage = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            FinalImage[hGap:hCal + hGap, :] = imgResize

    except ValueError as e:
        print(f"Error during image resizing: {e}")
        FinalImage = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    return FinalImage

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        handType = hand["type"]

        # Crop and resize the hand image
        FinalImage = crop_resize_img(lmList, imgSize, margin, handType)

        # Use the classifier to predict the gesture based on the cropped and resized image
        prediction, index = classifier.getPrediction(FinalImage)

        cv2.imshow("Cropped Hand", FinalImage)
        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
