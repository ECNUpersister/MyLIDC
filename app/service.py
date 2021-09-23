import os

import numpy as np
import torch

from model.unet.unet import UNet
from segmentation.transform import transform


def detect(img):
    img = np.array(img)
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.load_state_dict(torch.load(os.path.join('G:/MyLIDC/app/pth/unet/model.pth')))
    model = model.cuda()
    albumentations = True
    shape = 64
    img, _ = transform(albumentations, img, img, shape)
    img = img.unsqueeze(0).cuda()
    detect_img = model(img)
    detect_img = torch.sigmoid(detect_img).data.cpu().numpy()[0][0]
    detect_img = detect_img > 0.5
    return detect_img
