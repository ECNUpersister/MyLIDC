import math
import os
import sys
from glob import glob

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import transforms as T
import utils
from practice.dataset import MyDataset
from metric.metrics import dice_coeff

batch = 2


def get_maskrcnn(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if not math.isfinite(losses.item()):
            print(f"Loss is {losses.item()}, stopping training")
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])




def evaluate(model, data_loader_test, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    for images, targets in metric_logger.log_every(data_loader_test, 100, header):
        images = list(img.to(device) for img in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predicts = model(images)
        predict_mask = torch.zeros((batch, 1, 64, 64))
        index = 0
        for predict in predicts:
            masks = predict['masks']
            if masks.shape[0] != 0:
                for i in range(1, masks.shape[0]):
                    masks[0] = masks[0] + masks[i]
                predict_mask[index] = masks[0]
                index = index + 1
        index = 0
        gt_masks = torch.zeros((batch, 1, 64, 64))
        for target in targets:
            gt_mask = target['masks'].to(device)
            gt_masks[index] = gt_mask
            index = index + 1
        dice = dice_coeff(predict_mask, gt_masks)
        metric_logger.update(dice=dice)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    cur_path = 'G:/MyLIDC/data'
    dataset_name = 'lidc_shape64'
    train_image_path_list = glob(os.path.join('{}/dataset/{}/train/Image/*'.format(cur_path, dataset_name), "*.npy"))
    train_mask_path_list = glob(os.path.join('{}/dataset/{}/train/Mask/*'.format(cur_path, dataset_name), "*.npy"))
    val_image_path_list = glob(os.path.join('{}/dataset/{}/test/Image/*'.format(cur_path, dataset_name), "*.npy"))
    val_mask_path_list = glob(os.path.join('{}/dataset/{}/test/Mask/*'.format(cur_path, dataset_name), "*.npy"))
    dataset = MyDataset(train_image_path_list, train_mask_path_list, get_transform(train=True))
    dataset_test = MyDataset(val_image_path_list, val_mask_path_list, get_transform(train=False))
    # split the dataset in train and test set
    indices1 = torch.randperm(len(dataset)).tolist()
    indices2 = torch.randperm(len(dataset_test)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices1[:50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices2[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_maskrcnn(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=0.05, weight_decay=0.0005) # adam有时会出现nan导致蹦年优化
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == '__main__':
    main()
