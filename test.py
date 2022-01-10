import cv2
import matplotlib.pyplot as plt
import numpy as np

img=np.load('G:\MyLIDC\data\\false_positive\\false_positive_npy\\train\\0002_NI000_slice002_159_368.npy')
img2= np.load('G:\MyLIDC\data\predict_npy\\train\LIDC-IDRI-0002\\0002_PR000_slice002.npy')
imgk=np.load('G:\MyLIDC\data\dataset\lidc_shape512\\train\Mask\LIDC-IDRI-0002\\0002_MA000_slice002.npy')
img3=np.zeros((512,512),dtype='uint8')
img3[img2]=255
# print(img3[368,159])
# print(np.where(img2))
contours, _ = cv2.findContours(img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    Area = cv2.contourArea(contour)
    if Area > 10 and Area != 261121:  # 如果是这个值说明这张图里没有假阳性。是全黑的
        boundingrect = cv2.boundingRect(contour)
        print(boundingrect)
# img4= img3[127:191,336:400]
# plt.imshow(imgk)
# plt.show()
# print(img4[32,32])
# (array([171, 172, 364, 364, 365, 365, 365, 365, 365, 366, 366, 366, 366,
#        366, 367, 367, 367, 367, 367, 368, 368, 368, 368, 369, 369, 369,
#        369, 370, 370, 370, 370, 371, 371, 371, 371, 372, 372], dtype=int64), array([231, 231, 158, 159, 157, 158, 159, 160, 161, 157, 158, 159, 160,
#        161, 157, 158, 159, 160, 161, 158, 159, 160, 161, 158, 159, 160,
#        161, 157, 158, 159, 160, 157, 158, 159, 160, 158, 159], dtype=int64))