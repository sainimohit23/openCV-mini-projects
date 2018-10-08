from pyimagesearch.transform import four_point_transform
import numpy as np
import argparse
import cv2
import imutils
from skimage.filters import threshold_local

image = cv2.imread('images/page.jpg')

# Step 1 : Edge detection
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image, height=500)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 70, 200)



# Step 2 : Finding Contours
contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if imutils.is_cv2() else contours[1]
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

for c in contours:
    
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) == 4:
        found = approx
        break

# =============================================================================
# cv2.drawContours(image, [found], -1, (0,255,0))    
# cv2.imshow('im', image)
# =============================================================================


# Step 3 : Perspective transform and thresholding

warped = four_point_transform(orig, found.reshape(4, 2) * ratio)

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
