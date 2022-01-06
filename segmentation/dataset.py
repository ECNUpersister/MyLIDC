import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from segmentation.transform import transform


class MyLidcDataset(Dataset):
    def __init__(self, image_path, mask_path, Albumentation=False, shape=64):
        """
        IMAGES_PATHS: list of images paths ['./Images/0001_01_images.npy','./Images/0001_02_images.npy']
        MASKS_PATHS: list of masks paths ['./Masks/0001_01_masks.npy','./Masks/0001_02_masks.npy']
        """
        self.image_path = list(sorted(image_path))
        self.mask_path = list(sorted(mask_path))
        self.shape = shape
        self.albumentation = Albumentation

    def __getitem__(self, index):
        image = np.load(self.image_path[index])

        # Hu值转换大概在-1000到400之间，我们将其转换到0-255之间
        image = (image + 1000) / 1400 * 255
        image[image > 255] = 255
        image[image < 0] = 0
        image = image.astype(np.uint8)

        mask = np.load(self.mask_path[index])
        image, mask = transform(self.albumentation, image, mask, self.shape)
        return image, mask

    def __len__(self):
        return len(self.image_path)
