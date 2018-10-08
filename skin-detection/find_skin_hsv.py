# Finding the HSV range of my skin

import numpy as np
import cv2
from pyimagesearch import imutils


img = cv2.imread('my_pic.jpg')

checking_roi = cv2.rectangle(img, (120, 200), (260, 355), (0, 255, 0), thickness=2, lineType=cv2.LINE_8)

roi = img[200:355, 120:260]
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

cv2.imshow('my_img', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()


min_of_each_matrix = []
for mat in hsv:
    min_of_each_matrix.append(mat.min(axis=0))
    
    
min_of_each_matrix = np.array(min_of_each_matrix)
min_val = min_of_each_matrix.min(axis=0)


max_of_each_matrix = []
for mat in hsv:
    max_of_each_matrix.append(mat.max(axis=0))

max_of_each_matrix = np.array(max_of_each_matrix)
max_val = max_of_each_matrix.max(axis=0)
