import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class LidcClassificationDataset(Dataset):
    def __init__(self, image_path_list):
        """
        IMAGES_PATHS: list of images paths ['./Images/0001_01_images.npy','./Images/0001_02_images.npy']
        MASKS_PATHS: list of masks paths ['./Masks/0001_01_masks.npy','./Masks/0001_02_masks.npy']
        """
        self.image_path_list = image_path_list

    def __getitem__(self, index):
        image_name = self.image_path_list[index]
        image = np.load(image_name)
        label = 0
        if image_name[-8] != '_':  # 说明不是假阳性,假阳性已经做过Hu值裁剪了
            # Hu值转换大概在-1000到400之间，我们将其转换到0-255之间
            image = (image + 1000) / 1400 * 255
            image[image > 255] = 255
            image[image < 0] = 0
            label = 1  # 只能写1 不能写 1.0 因为 1.0 是double类型，要求long类型
        image = image.astype(np.uint8)
        transformations = transforms.Compose([transforms.ToTensor()])
        image = transformations(image)
        image = image.type(torch.FloatTensor)
        return image, label

    def __len__(self):
        return len(self.image_path_list)
