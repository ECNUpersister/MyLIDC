import cv2
import numpy as np

img=np.zeros((512,512),dtype='uint8')
img[:,:]=255
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    print(cv2.contourArea(contour)==261121)