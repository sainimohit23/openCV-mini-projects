import cv2


class eyeTracker: 
    
    def __init__(self, facePath, eyePath):
        self.faceCascade = cv2.CascadeClassifier(facePath)
        self.eyeCascade = cv2.CascadeClassifier(eyePath)
        
    def track(self, image):
            
        faceRects =  self.faceCascade.detectMultiScale(image, scaleFactor= 1.02,
                                                   minNeighbors= 15, minSize= (100, 100),
                                                   flags= cv2.CASCADE_SCALE_IMAGE)
        
        rects = []
        
        for (fx, fy, fw, fh) in faceRects:
            
            faceROI = image[fy:fy+fh, fx:fx+fw]
            rects.append((fx, fy, fx+fw, fy+fh))
            eyeRects = self.eyeCascade.detectMultiScale(faceROI, scaleFactor= 1.02,
                                                   minNeighbors= 20, minSize= (30, 30),
                                                   flags= cv2.CASCADE_SCALE_IMAGE)
        
            for (ex, ey, ew, eh) in eyeRects:
                rects.append((fx+ex, fy+ey, fx+ex+ew, fy+ey+eh))
                
        
        
        return rects





