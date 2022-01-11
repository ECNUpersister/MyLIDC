import os
from glob import glob

import numpy as np
import torch
from PIL import Image

from model.segmentation.unet.unet import UNet
from segmentation.transform import transform

cur_path = 'G:/MyLIDC/data'
dataset = 'lidc_shape512'
augmentations = True
shape = 512
model = UNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load('G:/MyLIDC/app/pth/unet/model.pth'))
device = torch.device('cuda')
model = model.to(device)
dir_pre_npy = '/data/predict_npy/test/'
dir_pre_png = '/data/predict_png/test/'

"""
这个工程使用来跑第一阶段粗定位产生的预测结果，用于制作第二阶段假阳性筛除的数据集
"""


def main():
    image_path_list = glob(os.path.join('{}/dataset/{}/test/Image/*'.format(cur_path, dataset), "*.npy"))
    image_path_list = list(sorted(image_path_list))
    length = len(image_path_list)
    for index in range(length):
        img_index_path = image_path_list[index]
        img_name = img_index_path[-23:]  # 差不多长这样 0001_NI000_slice000.npy
        img_name_npy = img_name.replace("NI", "PR")  # NI:Nodule,PR:Predict
        img_name_png = img_name_npy.replace("npy", "png")
        patient_id = 'LIDC-IDRI-' + img_name[:4]  # 差不多长这样 LIDC-IDRI-0001
        image = np.load(img_index_path)
        image = (image + 1000) / 1400 * 255
        image[image > 255] = 255
        image[image < 0] = 0
        image = image.astype(np.uint8)
        image, _ = transform(augmentations, image, image, shape)
        image = image.unsqueeze(0).to(device)
        detect_img = model(image)
        detect_img = torch.sigmoid(detect_img).data.cpu().numpy()[0][0]
        detect_img = detect_img > 0.5
        dir_npy = dir_pre_npy + patient_id
        dir_png = dir_pre_png + patient_id
        if not os.path.exists(dir_npy):
            os.makedirs(dir_npy)
        np.save(dir_npy + '/' + img_name_npy, detect_img)
        if not os.path.exists(dir_png):
            os.makedirs(dir_png)
        detect_img_png = Image.fromarray(detect_img)
        detect_img_png.save(dir_png + '/' + img_name_png)
        print("第{}张图像已执行".format(index))


if __name__ == '__main__':
    main()
