import numpy as np
import cv2
from FaceDetector import faceDetector


image = cv2.imread('pic.jpg')

r = 1080 / image.shape[1]
dim = (1080, int(image.shape[0] * r))


image = cv2.resize(image, dim, cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fd = faceDetector('models/haarcascade_frontalface_default.xml')

facerects = fd.detect(gray)

for (x, y, w, h) in facerects:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
cv2.imshow('detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()