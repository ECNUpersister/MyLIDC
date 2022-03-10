import os
import numpy as np
import torch
from PIL import Image


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_path_list, mask_path_list, transforms):
        self.image_path_list = sorted(image_path_list)
        self.mask_path_list = sorted(mask_path_list)
        self.transforms = transforms

    def __getitem__(self, idx):
        # load images and masks
        if self.image_path_list[idx].endswith('npy'):
            img = np.load(self.image_path_list[idx])
            img = Image.fromarray(img)
            mask = np.load(self.mask_path_list[idx])
        else:  # 如果是png或者jpg文件
            img = Image.open(self.image_path_list[idx])
            mask = Image.open(self.mask_path_list[idx])
            mask = np.array(mask)
        # img = img.convert("RGB")
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        boxes = []
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((1,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_path_list)
