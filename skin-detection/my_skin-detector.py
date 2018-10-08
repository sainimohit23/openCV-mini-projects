import numpy as np
import cv2
from pyimagesearch import imutils


camera = cv2.VideoCapture(0)
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([170, 255, 255], dtype = "uint8")


while True:
    
    ret, frame = camera.read()
    
    frame = imutils.resize(frame, width = 400)
    hsv_cvted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinmask = cv2.inRange(hsv_cvted, lower, upper)
    
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinmask = cv2.erode(skinmask, kernal, iterations=2)
    skinmask = cv2.dilate(skinmask, kernal, iterations=2)
    
    
    
    skinmask = cv2.GaussianBlur(skinmask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinmask)
    cv2.imshow("images", np.hstack([frame, skin]))
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()