import os
import torch
import numpy as np
from model.unet.unet import UNet
from segmentation.transform import transform


def detect(img):
    # 输入png图像是512*512*4 RGBA四通道图像，从前三个通道任取一个即可
    img = np.array(img, dtype='uint8')[:, :, 0]
    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(os.path.join('G:/MyLIDC/app/pth/unet/model.pth')))
    model = model.cuda()
    albumentations = True
    shape = 512
    img, _ = transform(albumentations, img, img, shape)
    img = img.unsqueeze(0).cuda()
    detect_img = model(img)
    detect_img = torch.sigmoid(detect_img).data.cpu().numpy()[0][0]
    detect_img = detect_img > 0.5
    return detect_img
