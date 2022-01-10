import math
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image

"""
这个工程使用来跑第一阶段粗定位产生的预测结果，用于制作第二阶段假阳性筛除的数据集
"""


def main():
    predict_path_list = glob(os.path.join('data/predict_npy/train/*', "*.npy"))  # 载入网络一预测结果列表
    mask_path_list = glob(os.path.join('data/dataset/lidc_shape512/train/Mask/*', "*.npy"))  # 载入GT Mask列表
    img_path_list = glob(os.path.join('data/dataset/lidc_shape512/train/Image/*', "*.npy"))
    predict_path_list = list(sorted(predict_path_list))
    mask_path_list = list(sorted(mask_path_list))
    img_path_list = list(sorted(img_path_list))
    length = len(predict_path_list)
    for index in range(length):
        mask_index_path = mask_path_list[index]
        predict_index_path = predict_path_list[index]
        img_index_path = img_path_list[index]
        img_name = img_index_path[-23:-4]  # 差不多长这样 0001_NI000_slice000
        mask = np.load(mask_index_path)
        pos = np.where(mask)
        xmin = np.min(pos[0])
        xmax = np.max(pos[0])
        ymin = np.min(pos[1])
        ymax = np.max(pos[1])
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        xmin = 0 if x - 32 < 0 else math.floor(x - 32)
        xmax = 511 if x + 32 > 511 else math.ceil(x + 32)
        ymin = 0 if y - 32 < 0 else math.floor(y - 32)
        ymax = 511 if y + 32 > 511 else math.ceil(y + 32)
        predic = np.load(predict_index_path)
        img = np.load(img_index_path)
        # 以下是图像Hu值修正固定流程
        img = (img + 1000) / 1400 * 255
        img[img > 255] = 255
        img[img < 0] = 0
        img = img.astype(np.uint8)

        predic[xmin:xmax + 1, ymin:ymax + 1] = False  # 根据GT Mask制作遮罩，将predict中的真预测遮住，那剩下的就全都是假预测了
        predict_uint8 = np.zeros((512, 512), dtype='uint8')
        predict_uint8[np.where(predic)] = 255
        contours, hierarchy = cv2.findContours(predict_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            for contour in contours:
                if cv2.contourArea(contour) > 10:
                    boundingrect = cv2.boundingRect(contour)
                    centerx = boundingrect[0] + math.floor(boundingrect[2] / 2)
                    centery = boundingrect[1] + math.floor(boundingrect[3] / 2)
                    position = make_slice64(centerx, centery)
                    false_positive_img_shape64 = img[position[0]:position[1] + 1, position[2]:position[3] + 1]
                    false_positive_mask_shape64 = predict_uint8[position[0]:position[1] + 1,
                                                  position[2]:position[3] + 1]
                    false_positive_img_shape64_name = img_name + '_' + str(centerx) + '_' + str(centery)
                    false_positive_img_shape64_name_npy = false_positive_img_shape64_name + '.npy'
                    false_positive_img_shape64_name_png = false_positive_img_shape64_name + '.png'
                    np.save('data/false_positive/false_positive_npy/train/' + false_positive_img_shape64_name_npy,
                            false_positive_img_shape64)
                    false_positive_img_shape64 = Image.fromarray(false_positive_img_shape64)
                    false_positive_img_shape64.save(
                        'data/false_positive/false_positive_png/train/' + false_positive_img_shape64_name_png)
                    false_positive_mask_shape64 = Image.fromarray(false_positive_mask_shape64)
                    false_positive_mask_shape64.save(
                        'data/false_positive/false_positive_mask_png/train/' + false_positive_img_shape64_name_png)
        print("第{}张图像已执行".format(index))


def make_slice64(x, y):
    xmin = 0 if x - 32 < 0 else x - 32
    ymin = 0 if y - 32 < 0 else y - 32
    if x + 32 > 511:
        xmin = 448
        xmax = 511
    else:
        xmax = xmin + 63
    if y + 32 > 511:
        ymin = 448
        ymax = 511
    else:
        ymax = ymin + 63
    return (xmin, xmax, ymin, ymax)


if __name__ == '__main__':
    main()
