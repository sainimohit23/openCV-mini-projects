import numpy as np
import cv2
from eyeTracker import eyeTracker
import imutils

camera = cv2.VideoCapture(0)
et = eyeTracker('models/haarcascade_frontalface_default.xml', 'models/haarcascade_eye.xml')

while True:
    ret, frame = camera.read()
    frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = et.track(gray)
    
    for rect in rects:
        cv2.rectangle(frame, (rect[0], rect[1]),
                              (rect[2], rect[3]), (0, 255, 0), 2)
        
    cv2.imshow("Tracking", frame)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()