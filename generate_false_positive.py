import math
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from model.unet.unet import UNet
from segmentation.transform import transform

cur_path = 'G:/MyLIDC/data'
dataset = 'lidc_shape512'
augmentations = True
shape = 512
model = UNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load('G:/MyLIDC/app/pth/unet/model.pth'))
device = torch.device('cuda')
model = model.to(device)
dir_pre_npy = '/data/predict_npy/train/'
dir_pre_png = '/data/predict_png/train/'

"""
这个工程使用来跑第一阶段粗定位产生的预测结果，用于制作第二阶段假阳性筛除的数据集
"""


def main():
    image_path_list = glob(os.path.join('{}/dataset/{}/train/Image/*'.format(cur_path, dataset), "*.npy"))
    mask_path_list = glob(os.path.join('{}/dataset/{}/train/Mask/*'.format(cur_path, dataset), "*.npy"))
    image_path_list = list(sorted(image_path_list))
    length = len(image_path_list)
    for index in range(length):
        mask_index_path = mask_path_list[index]
        mask = np.load(mask_index_path)


if __name__ == '__main__':
    # main()
    # image_path_list = glob(os.path.join('{}/dataset/{}/train/Image/*'.format(cur_path, dataset), "*.npy"))
    # mask_path_list = glob(os.path.join('{}/dataset/{}/train/Mask/*'.format(cur_path, dataset), "*.npy"))
    # mask_index_path = mask_path_list[0]
    # mask = np.load(mask_index_path)
    mask2 = np.load('G:\MyLIDC\data\predict_npy\\train\LIDC-IDRI-0003\\0003_PR003_slice003.npy')
    mask =np.load('G:\MyLIDC\data\dataset\lidc_shape512\\train\Mask\LIDC-IDRI-0003\\0003_MA003_slice003.npy')
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
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(mask2,cmap=plt.cm.gray)
    mask2[xmin:xmax + 1, ymin:ymax + 1]=False
    ax[1].imshow(mask2, cmap=plt.cm.gray)
    ax[2].imshow(mask, cmap=plt.cm.gray)
    plt.show()
    # print(pos)
