import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import math

directory1 = 'Da/'
directory2 = 'Te/'
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 224
margin = 20  # Margin to prevent points from being cut off
maxSamples = 150
maxTest = maxSamples + math.ceil(maxSamples / 4)

# Define colors for each finger
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

# Wrist point index in landmark list (0) and fingertip indexes
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
    count = {
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
             }
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        lmList = hand['lmList']  # List of 21 Landmark points
        handType = hand['type']  # 'Left' or 'Right'

        FinalImage = crop_resize_img(lmList, imgSize, margin, handType)
        cv2.imshow("FinalImage", FinalImage)

    cv2.imshow("Image", img)
    interrupt = cv2.waitKey(5)

    if interrupt & 0xFF == ord('a') and count['a'] < maxSamples:
        cv2.imwrite(directory1+'A/'+'A-'+str(count['a'])+'.jpg',FinalImage)
        print(count['a'])
    elif interrupt & 0xFF == ord('a') and count['a'] < maxTest:
        cv2.imwrite(directory2+'A/'+'A-'+str(count['a'])+'.jpg',FinalImage)
        print(count['a'])

    elif interrupt & 0xFF == ord('b') and count['b'] < maxSamples:
        cv2.imwrite(directory1+'B/'+'B-'+str(count['b'])+'.jpg',FinalImage)
        print(count['b'])
    elif interrupt & 0xFF == ord('b') and count['b'] < maxTest:
        cv2.imwrite(directory2+'B/'+'B-'+str(count['b'])+'.jpg',FinalImage)
        print(count['b'])

    elif interrupt & 0xFF == ord('c') and count['c'] < maxSamples:
        cv2.imwrite(directory1+'C/'+'C-'+str(count['c'])+'.jpg',FinalImage)
        print(count['c'])
    elif interrupt & 0xFF == ord('c') and count['c'] < maxTest:
        cv2.imwrite(directory2+'C/'+'C-'+str(count['c'])+'.jpg',FinalImage)
        print(count['c'])

    elif interrupt & 0xFF == ord('d') and count['d'] < maxSamples:
        cv2.imwrite(directory1+'D/'+'D-'+str(count['d'])+'.jpg',FinalImage)
        print(count['d'])
    elif interrupt & 0xFF == ord('d') and count['d'] < maxTest:
        cv2.imwrite(directory2+'D/'+'D-'+str(count['d'])+'.jpg',FinalImage)
        print(count['d'])

    elif interrupt & 0xFF == ord('e') and count['e'] < maxSamples:
        cv2.imwrite(directory1+'E/'+'E-'+str(count['e'])+'.jpg',FinalImage)
        print(count['e'])
    elif interrupt & 0xFF == ord('e') and count['e'] < maxTest:
        cv2.imwrite(directory2+'E/'+'E-'+str(count['e'])+'.jpg',FinalImage)
        print(count['e'])

    elif interrupt & 0xFF == ord('f') and count['f'] < maxSamples:
        cv2.imwrite(directory1+'F/'+'F-'+str(count['f'])+'.jpg',FinalImage)
        print(count['f'])
    elif interrupt & 0xFF == ord('f') and count['f'] < maxTest:
        cv2.imwrite(directory2+'F/'+'F-'+str(count['f'])+'.jpg',FinalImage)
        print(count['f'])

    elif interrupt & 0xFF == ord('g') and count['g'] < maxSamples:
        cv2.imwrite(directory1+'G/'+'G-'+str(count['g'])+'.jpg',FinalImage)
        print(count['g'])
    elif interrupt & 0xFF == ord('g') and count['g'] < maxTest:
        cv2.imwrite(directory2+'G/'+'G-'+str(count['g'])+'.jpg',FinalImage)
        print(count['g'])

    elif interrupt & 0xFF == ord('h') and count['h'] < maxSamples:
        cv2.imwrite(directory1+'H/'+'H-'+str(count['h'])+'.jpg',FinalImage)
        print(count['h'])
    elif interrupt & 0xFF == ord('h') and count['h'] < maxTest:
        cv2.imwrite(directory2+'H/'+'H-'+str(count['h'])+'.jpg',FinalImage)
        print(count['h'])

    elif interrupt & 0xFF == ord('i') and count['i'] < maxSamples:
        cv2.imwrite(directory1+'I/'+'I-'+str(count['i'])+'.jpg',FinalImage)
        print(count['i'])
    elif interrupt & 0xFF == ord('i') and count['i'] < maxTest:
        cv2.imwrite(directory2+'I/'+'I-'+str(count['i'])+'.jpg',FinalImage)
        print(count['i'])

    elif interrupt & 0xFF == ord('j') and count['j'] < maxSamples:
        cv2.imwrite(directory1+'J/'+'J-'+str(count['j'])+'.jpg',FinalImage)
        print(count['j'])
    elif interrupt & 0xFF == ord('j') and count['j'] < maxTest:
        cv2.imwrite(directory2+'J/'+'J-'+str(count['j'])+'.jpg',FinalImage)
        print(count['j'])

    elif interrupt & 0xFF == ord('k') and count['k'] < maxSamples:
        cv2.imwrite(directory1+'K/'+'K-'+str(count['k'])+'.jpg',FinalImage)
        print(count['k'])
    elif interrupt & 0xFF == ord('k') and count['k'] < maxTest:
        cv2.imwrite(directory2+'K/'+'K-'+str(count['k'])+'.jpg',FinalImage)
        print(count['k'])

    elif interrupt & 0xFF == ord('l') and count['l'] < maxSamples:
        cv2.imwrite(directory1+'L/'+'L-'+str(count['l'])+'.jpg',FinalImage)
        print(count['l'])
    elif interrupt & 0xFF == ord('l') and count['l'] < maxTest:
        cv2.imwrite(directory2+'L/'+'L-'+str(count['l'])+'.jpg',FinalImage)
        print(count['l'])
    
    elif interrupt & 0xFF == ord('m') and count['m'] < maxSamples:
        cv2.imwrite(directory1+'M/'+'M-'+str(count['m'])+'.jpg',FinalImage)
        print(count['m'])
    elif interrupt & 0xFF == ord('m') and count['m'] < maxTest:
        cv2.imwrite(directory2+'M/'+'M-'+str(count['m'])+'.jpg',FinalImage)
        print(count['m'])

    elif interrupt & 0xFF == ord('n') and count['n'] < maxSamples:
        cv2.imwrite(directory1+'N/'+'N-'+str(count['n'])+'.jpg',FinalImage)
        print(count['n'])
    elif interrupt & 0xFF == ord('n') and count['n'] < maxTest:
        cv2.imwrite(directory2+'N/'+'N-'+str(count['n'])+'.jpg',FinalImage)
        print(count['n'])

    elif interrupt & 0xFF == ord('o') and count['o'] < maxSamples:
        cv2.imwrite(directory1+'O/'+'O-'+str(count['o'])+'.jpg',FinalImage)
        print(count['o'])
    elif interrupt & 0xFF == ord('o') and count['o'] < maxTest:
        cv2.imwrite(directory2+'O/'+'O-'+str(count['o'])+'.jpg',FinalImage)
        print(count['o'])

    elif interrupt & 0xFF == ord('p') and count['p'] < maxSamples:
        cv2.imwrite(directory1+'P/'+'P-'+str(count['p'])+'.jpg',FinalImage)
        print(count['p'])
    elif interrupt & 0xFF == ord('p') and count['p'] < maxTest:
        cv2.imwrite(directory2+'P/'+'P-'+str(count['p'])+'.jpg',FinalImage)
        print(count['p'])

    elif interrupt & 0xFF == ord('q') and count['q'] < maxSamples:
        cv2.imwrite(directory1+'Q/'+'Q-'+str(count['q'])+'.jpg',FinalImage)
        print(count['q'])
    elif interrupt & 0xFF == ord('q') and count['q'] < maxTest:
        cv2.imwrite(directory2+'Q/'+'Q-'+str(count['q'])+'.jpg',FinalImage)
        print(count['q'])

    elif interrupt & 0xFF == ord('r') and count['r'] < maxSamples:
        cv2.imwrite(directory1+'R/'+'R-'+str(count['r'])+'.jpg',FinalImage)
        print(count['r'])
    elif interrupt & 0xFF == ord('r') and count['r'] < maxTest:
        cv2.imwrite(directory2+'R/'+'R-'+str(count['r'])+'.jpg',FinalImage)
        print(count['r'])

    elif interrupt & 0xFF == ord('s') and count['s'] < maxSamples:
        cv2.imwrite(directory1+'S/'+'S-'+str(count['s'])+'.jpg',FinalImage)
        print(count['s'])
    elif interrupt & 0xFF == ord('s') and count['s'] < maxTest:
        cv2.imwrite(directory2+'S/'+'S-'+str(count['s'])+'.jpg',FinalImage)
        print(count['s'])

    elif interrupt & 0xFF == ord('t') and count['t'] < maxSamples:
        cv2.imwrite(directory1+'T/'+'T-'+str(count['t'])+'.jpg',FinalImage)
        print(count['t'])
    elif interrupt & 0xFF == ord('t') and count['t'] < maxTest:
        cv2.imwrite(directory2+'T/'+'T-'+str(count['t'])+'.jpg',FinalImage)
        print(count['t'])

    elif interrupt & 0xFF == ord('u') and count['u'] < maxSamples:
        cv2.imwrite(directory1+'U/'+'U-'+str(count['u'])+'.jpg',FinalImage)
        print(count['u'])
    elif interrupt & 0xFF == ord('u') and count['u'] < maxTest:
        cv2.imwrite(directory2+'U/'+'U-'+str(count['u'])+'.jpg',FinalImage)
        print(count['u'])

    elif interrupt & 0xFF == ord('v') and count['v'] < maxSamples:
        cv2.imwrite(directory1+'V/'+'V-'+str(count['v'])+'.jpg',FinalImage)
        print(count['v'])
    elif interrupt & 0xFF == ord('v') and count['v'] < maxTest:
        cv2.imwrite(directory2+'V/'+'V-'+str(count['v'])+'.jpg',FinalImage)
        print(count['v'])

    elif interrupt & 0xFF == ord('w') and count['w'] < maxSamples:
        cv2.imwrite(directory1+'W/'+'W-'+str(count['w'])+'.jpg',FinalImage)
        print(count['w'])
    elif interrupt & 0xFF == ord('w') and count['w'] < maxTest:
        cv2.imwrite(directory2+'W/'+'W-'+str(count['w'])+'.jpg',FinalImage)
        print(count['w'])

    elif interrupt & 0xFF == ord('x') and count['x'] < maxSamples:
        cv2.imwrite(directory1+'X/'+'X-'+str(count['x'])+'.jpg',FinalImage)
        print(count['x'])
    elif interrupt & 0xFF == ord('x') and count['x'] < maxTest:
        cv2.imwrite(directory2+'X/'+'X-'+str(count['x'])+'.jpg',FinalImage)
        print(count['x'])

    elif interrupt & 0xFF == ord('y') and count['y'] < maxSamples:
        cv2.imwrite(directory1+'Y/'+'Y-'+str(count['y'])+'.jpg',FinalImage)
        print(count['y'])
    elif interrupt & 0xFF == ord('y') and count['y'] < maxTest:
        cv2.imwrite(directory2+'Y/'+'Y-'+str(count['y'])+'.jpg',FinalImage)
        print(count['y'])

    elif interrupt & 0xFF == ord('z') and count['z'] < maxSamples:
        cv2.imwrite(directory1+'Z/'+'Z-'+str(count['z'])+'.jpg',FinalImage)
        print(count['z'])
    elif interrupt & 0xFF == ord('z') and count['z'] < maxTest:
        cv2.imwrite(directory2+'Z/'+'Z-'+str(count['z'])+'.jpg',FinalImage)
        print(count['z'])
