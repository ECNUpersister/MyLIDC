import numpy as np
import torch

from model.segmentation.unet.unet import UNet
from lidc_segmentation.transform import transform


def detect(img):
    # 输入png图像是512*512*4 RGBA四通道图像，从前三个通道任取一个即可
    img = np.array(img, dtype='uint8')[:, :, 0]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load('G:/MyLIDC/app/pth/unet/model.pth'))
    model = model.to(device)
    albumentations = True
    shape = 512
    img, _ = transform(albumentations, img, img, shape)
    img = img.unsqueeze(0).to(device)
    detect_img = model(img)
    detect_img = torch.sigmoid(detect_img).data.cpu().numpy()[0][0]
    detect_img = detect_img > 0.2
    return detect_img
