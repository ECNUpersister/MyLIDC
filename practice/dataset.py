import numpy as np
import torch
from PIL import Image


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_path_list, mask_path_list, transforms):
        self.image_path_list = sorted(image_path_list)
        self.mask_path_list = sorted(mask_path_list)
        self.transforms = transforms

    def __getitem__(self, idx):
        if self.image_path_list[idx].endswith('npy'):
            img = np.load(self.image_path_list[idx])
            img = Image.fromarray(img)
            mask = np.load(self.mask_path_list[idx])
        else:  # 如果是png或者jpg文件
            img = Image.open(self.image_path_list[idx])
            mask = Image.open(self.mask_path_list[idx])
            mask = np.array(mask)
        pos = np.where(mask)
        boxes = torch.as_tensor([[np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)
        masks = torch.as_tensor(mask[np.newaxis, :, :], dtype=torch.uint8)
        target = {"boxes": boxes, "labels": labels, "masks": masks}
        img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_path_list)
