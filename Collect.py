import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import math
import string

directory1 = 'Data/'
directory2 = 'Test/'
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 224
margin = 20
maxSamples = 800
maxTest = maxSamples + math.ceil(maxSamples / 4)

if not os.path.exists(directory1):
    os.makedirs(directory1)

if not os.path.exists(directory2):
    os.makedirs(directory2)

for char in string.ascii_uppercase:
    subdir1 = os.path.join(directory1, char)
    subdir2 = os.path.join(directory2, char)

    if not os.path.exists(subdir1):
        os.makedirs(subdir1)

    if not os.path.exists(subdir2):
        os.makedirs(subdir2)

if not os.path.exists(directory1 + '_'):
    os.makedirs(directory1 + '_')
if not os.path.exists(directory2 + '_'):
    os.makedirs(directory2 + '_')

if not os.path.exists(directory1 + 'del'):
    os.makedirs(directory1 + 'del')
if not os.path.exists(directory2 + 'del'):
    os.makedirs(directory2 + 'del')

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

while True:
    count = {

        # Count number of samples collected for each letter
    #count = {letter: len(os.listdir(f'{directory1}{letter}')) + len(os.listdir(f'{directory2}{letter}'))
          #   for letter in string.ascii_uppercase + '_='}

             'a': len(os.listdir(directory1+"/A"))+len(os.listdir(directory2+"/A")),
             'b': len(os.listdir(directory1+"/B"))+len(os.listdir(directory2+"/B")),
             'c': len(os.listdir(directory1+"/C"))+len(os.listdir(directory2+"/C")),
             'd': len(os.listdir(directory1+"/D"))+len(os.listdir(directory2+"/D")),
             'e': len(os.listdir(directory1+"/E"))+len(os.listdir(directory2+"/E")),
             'f': len(os.listdir(directory1+"/F"))+len(os.listdir(directory2+"/F")),
             'g': len(os.listdir(directory1+"/G"))+len(os.listdir(directory2+"/G")),
             'h': len(os.listdir(directory1+"/H"))+len(os.listdir(directory2+"/H")),
             'i': len(os.listdir(directory1+"/I"))+len(os.listdir(directory2+"/I")),
             'j': len(os.listdir(directory1+"/J"))+len(os.listdir(directory2+"/J")),
             'k': len(os.listdir(directory1+"/K"))+len(os.listdir(directory2+"/K")),
             'l': len(os.listdir(directory1+"/L"))+len(os.listdir(directory2+"/L")),
             'm': len(os.listdir(directory1+"/M"))+len(os.listdir(directory2+"/M")),
             'n': len(os.listdir(directory1+"/N"))+len(os.listdir(directory2+"/N")),
             'o': len(os.listdir(directory1+"/O"))+len(os.listdir(directory2+"/O")),
             'p': len(os.listdir(directory1+"/P"))+len(os.listdir(directory2+"/P")),
             'q': len(os.listdir(directory1+"/Q"))+len(os.listdir(directory2+"/Q")),
             'r': len(os.listdir(directory1+"/R"))+len(os.listdir(directory2+"/R")),
             's': len(os.listdir(directory1+"/S"))+len(os.listdir(directory2+"/S")),
             't': len(os.listdir(directory1+"/T"))+len(os.listdir(directory2+"/T")),
             'u': len(os.listdir(directory1+"/U"))+len(os.listdir(directory2+"/U")),
             'v': len(os.listdir(directory1+"/V"))+len(os.listdir(directory2+"/V")),
             'w': len(os.listdir(directory1+"/W"))+len(os.listdir(directory2+"/W")),
             'x': len(os.listdir(directory1+"/X"))+len(os.listdir(directory2+"/X")),
             'y': len(os.listdir(directory1+"/Y"))+len(os.listdir(directory2+"/Y")),
             'z': len(os.listdir(directory1+"/Z"))+len(os.listdir(directory2+"/Z")),
             '_': len(os.listdir(directory1+"/_"))+len(os.listdir(directory2+"/_")),
             '=': len(os.listdir(directory1+"/del"))+len(os.listdir(directory2+"/del")),
             }

    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        lmList = hand['lmList']  
        handType = hand['type'] 

        FinalImage = crop_resize_img(lmList, imgSize, margin, handType)
        cv2.imshow("FinalImage", FinalImage)

    cv2.imshow("Image", img)
    interrupt = cv2.waitKey(5)

    for letter in 'abcdefghijklmnopqrstuvwxyz':
        if interrupt & 0xFF == ord(letter) and count[letter] < maxSamples:
            cv2.imwrite(f'{directory1}{letter.upper()}/{letter.upper()}-{count[letter]}.jpg', FinalImage)
            print(count[letter])
        elif interrupt & 0xFF == ord(letter) and count[letter] < maxTest:
            cv2.imwrite(f'{directory2}{letter.upper()}/{letter.upper()}-{count[letter]}.jpg', FinalImage)
            print(count[letter])
     
    if interrupt & 0xFF == ord('-') and count['_'] < maxSamples:
        cv2.imwrite(f'{directory1}_/_-{count["_"]}.jpg', FinalImage)
        print(count["_"])

    elif interrupt & 0xFF == ord('-') and count['_'] < maxTest:
        cv2.imwrite(f'{directory2}_/_-{count["_"]}.jpg', FinalImage)
        print(count["_"])

    if interrupt & 0xFF == ord('=') and count['='] < maxSamples:
        cv2.imwrite(f'{directory1}del/-{count["="]}.jpg', FinalImage)
        print(count["="])

    elif interrupt & 0xFF == ord('=') and count['='] < maxTest:
        cv2.imwrite(f'{directory2}del/-{count["="]}.jpg', FinalImage)
        print(count["="])

    if interrupt & 0xFF == 27:
        break

cv2.destroyAllWindows()
