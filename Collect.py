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
    
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'A/'+str(count['a'])+'.jpg',FinalImage)
        print(count['a'])
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'B/'+str(count['b'])+'.jpg',FinalImage)
        print(count['b'])
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'C/'+str(count['c'])+'.jpg',FinalImage)
        print(count['c'])
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'D/'+str(count['d'])+'.jpg',FinalImage)
        print(count['d'])
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'E/'+str(count['e'])+'.jpg',FinalImage)
        print(count['e'])
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory+'F/'+str(count['f'])+'.jpg',FinalImage)
        print(count['f'])
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'G/'+str(count['g'])+'.jpg',FinalImage)
        print(count['g'])
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'H/'+str(count['h'])+'.jpg',FinalImage)
        print(count['h'])
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'I/'+str(count['i'])+'.jpg',FinalImage)
        print(count['i'])
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'J/'+str(count['j'])+'.jpg',FinalImage)
        print(count['j'])
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory+'K/'+str(count['k'])+'.jpg',FinalImage)
        print(count['k'])
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'L/'+str(count['l'])+'.jpg',FinalImage)
        print(count['l'])
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory+'M/'+str(count['m'])+'.jpg',FinalImage)
        print(count['m'])
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'N/'+str(count['n'])+'.jpg',FinalImage)
        print(count['n'])
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory+'O/'+str(count['o'])+'.jpg',FinalImage)
        print(count['o'])
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory+'P/'+str(count['p'])+'.jpg',FinalImage)
        print(count['p'])
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory+'Q/'+str(count['q'])+'.jpg',FinalImage)
        print(count['q'])
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'R/'+str(count['r'])+'.jpg',FinalImage)
        print(count['r'])
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'S/'+str(count['s'])+'.jpg',FinalImage)
        print(count['s'])
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'T/'+str(count['t'])+'.jpg',FinalImage)
        print(count['t'])
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory+'U/'+str(count['u'])+'.jpg',FinalImage)
        print(count['u'])
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory+'V/'+str(count['v'])+'.jpg',FinalImage)
        print(count['v'])
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory+'W/'+str(count['w'])+'.jpg',FinalImage)
        print(count['w'])
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory+'X/'+str(count['x'])+'.jpg',FinalImage)
        print(count['x'])
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory+'Y/'+str(count['y'])+'.jpg',FinalImage)
        print(count['y'])
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory+'Z/'+str(count['z'])+'.jpg',FinalImage)
        print(count['z'])
