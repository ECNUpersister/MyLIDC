import math
import os
import sys

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Subset
from metric.metrics import dice_coeff
from practice.dataset import MyDataset
from utils import *
import transforms as T
from glob import glob


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_dataset(dataset_name, sample_ratio):
    train_image_path_list = glob(os.path.join('{}/train/Image/*'.format(dataset_name), "*.npy"))
    train_mask_path_list = glob(os.path.join('{}/train/Mask/*'.format(dataset_name), "*.npy"))
    val_image_path_list = glob(os.path.join('{}/test/Image/*'.format(dataset_name), "*.npy"))
    val_mask_path_list = glob(os.path.join('{}/test/Mask/*'.format(dataset_name), "*.npy"))
    dataset_train, dataset_test = MyDataset(train_image_path_list, train_mask_path_list, get_transform(train=True)), \
                                  MyDataset(val_image_path_list, val_mask_path_list, get_transform(train=False))
    len_train = len(dataset_train)
    len_test = len(dataset_test)
    indices_train = torch.randperm(len_train).tolist()
    indices_test = torch.randperm(len_test).tolist()
    dataset_train = Subset(dataset_train, indices_train[:math.floor(len_train * sample_ratio)])
    dataset_test = Subset(dataset_test, indices_test[:math.floor(len_test * sample_ratio)])
    return dataset_train, dataset_test


def get_maskrcnn(pretrained):
    # pretrain是在coco数据集上进行的预训练
    if pretrained:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        # 修改Faster R-CNN检测头和Mask R-CNN检测头最后的类别数
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced=256, num_classes=2)
    # 也可以不要预训练，直接训练
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=2)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # TODO:为什么会出现非数loss
        if not math.isfinite(losses.item()):
            print(f"Loss is {losses.item()}, stopping training")
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def evaluate(model, data_loader_test, device, shape):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    for images, targets in metric_logger.log_every(data_loader_test, 100, header):
        images = list(img.to(device) for img in images)
        predicts = model(images)
        cur_batch = len(images)  # 因为dataloader没有设置drop_last，因此cur_batch有可能与batch——size是不相等的
        predict_masks = torch.zeros((cur_batch, 1, shape, shape))
        gt_masks = torch.zeros((cur_batch, 1, shape, shape))
        for i in range(cur_batch):
            predict_masks[i] = torch.sum(predicts[i]['masks'], dim=0)
            gt_masks[i] = targets[i]['masks'].to(device)
        dice = dice_coeff(predict_masks, gt_masks)
        metric_logger.update(dice=dice)
