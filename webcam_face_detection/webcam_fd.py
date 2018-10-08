import numpy as np
import cv2
import imutils
import FaceDetector


camera = cv2.VideoCapture(0)
fd = faceDetector('models/haarcascade_frontalface_default.xml')

while True:
    ret, frame = camera.read()
    
    frame = imutils.resize(frame, width = 800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faceRects = fd.detect(gray)
    for (x, y, w, h) in faceRects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()
