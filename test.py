import math

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # main()
    # image_path_list = glob(os.path.join('{}/dataset/{}/train/Image/*'.format(cur_path, dataset), "*.npy"))
    # mask_path_list = glob(os.path.join('{}/dataset/{}/train/Mask/*'.format(cur_path, dataset), "*.npy"))
    # mask_index_path = mask_path_list[0]
    # mask = np.load(mask_index_path)
    mask = np.load('G:\MyLIDC\data\predict_npy\\train\LIDC-IDRI-0003\\0003_PR003_slice003.npy')
    ll = np.zeros((512, 512), dtype='uint8')
    ll[np.where(mask)] = 255
    plt.imshow(ll)
    plt.show()
    # print(ll)
    # img =ll
    # gray = cv2.cvtColor(ll, cv2.COLOR_BGR2GRAY)
    # gray=ll
    # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    img = np.zeros((512, 512, 3), dtype='uint8')
    img[:, :, 0] = ll
    img[:, :, 1] = ll
    img[:, :, 2] = ll
    contours, hierarchy = cv2.findContours(ll, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            boundingrect = cv2.boundingRect(contour)




    print(contours)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

    plt.imshow(img)
    plt.show()
