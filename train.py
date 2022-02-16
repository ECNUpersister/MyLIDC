import os
import time
from collections import OrderedDict
from glob import glob

from tqdm import tqdm

from metric.losses import DiceLoss
from metric.metrics import iou_score, dice_coef
from metric.result import *
from metric.utils import AverageMeter
from model.segmentation.fcn.fcn import *
from model.segmentation.unet.unet import UNet
from lidc_segmentation.dataset import MyLidcDataset
from lidc_segmentation.view_output import view_output

cur_path = 'data'
dataset = 'lidc_shape64'
shape = 64
epochs = 500
batch_size = 2
early_stopping = 300
num_workers = 0
learning_rate = 1e-5
momentum = 0.9
weight_decay = 1e-4
augmentations = True
backbone = 'none'
my_model = UNet(n_channels=1, n_classes=1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train(data_loader, model, criterion, isTrain, optimizer):
    # isTrain 是一个布尔值，用来判断是trian阶段还是val阶段
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    bar = tqdm(total=len(data_loader))
    for input, target in data_loader:
        batch_size = input.size(0)
        input = input.to(device)
        target = target.to(device)
        output = model(input)

        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)
        if isTrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_meters['loss'].update(loss.item(), batch_size)
        avg_meters['iou'].update(iou, batch_size)
        avg_meters['dice'].update(dice, batch_size)

        metric = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg)
        ])
        bar.set_postfix(metric)
        bar.update(1)
    bar.close()
    return metric


def config_save():
    # 以时间戳记录输出
    file_name = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    os.makedirs('{}/model_outputs/{}'.format(cur_path, file_name))
    print("按照时间戳创建文件夹：", file_name)

    # 记录参数信息
    f = open('{}/model_outputs/{}/config.txt'.format(cur_path, file_name), "w", encoding='utf-8')
    f.write('dataset={},\nshape={},\nepochs={},\nbatch_size={},'
            '\nearly_stopping={},\nnum_workers={},'
            '\nlearning_rate={},\nmomentum={},'
            '\nweight_decay={},\naugmentations={},\nbackbone={}'
            .format(dataset, shape, epochs, batch_size,
                    early_stopping, num_workers,
                    learning_rate, momentum,
                    weight_decay, augmentations, backbone))
    f.close()
    return file_name


def main():
    file_name = config_save()
    criterion = DiceLoss().to(device)
    # cudnn.benchmark = True
    model = my_model.to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    train_image_path_list = glob(os.path.join('{}/dataset/{}/train/Image/*'.format(cur_path, dataset), "*.npy"))
    train_mask_path_list = glob(os.path.join('{}/dataset/{}/train/Mask/*'.format(cur_path, dataset), "*.npy"))

    val_image_path_list = glob(os.path.join('{}/dataset/{}/test/Image/*'.format(cur_path, dataset), "*.npy"))
    val_mask_path_list = glob(os.path.join('{}/dataset/{}/test/Mask/*'.format(cur_path, dataset), "*.npy"))
    print('*' * 50)
    print('训练集共计{}张图片'.format(len(train_image_path_list)))
    print('验证集共计{}张图片'.format(len(val_image_path_list)))
    print('验证集：训练集={:2f}'.format(len(val_image_path_list) / len(train_image_path_list)))
    print('*' * 50)

    train_dataset = MyLidcDataset(train_image_path_list, train_mask_path_list, augmentations, shape)
    val_dataset = MyLidcDataset(val_image_path_list, val_mask_path_list, augmentations, shape)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers)

    log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'train_loss', 'train_iou',
                                          'train_dice', 'val_loss', 'val_iou', 'val_dice'])

    best_dice = 0
    trigger = 0
    for epoch in range(epochs):
        model.train()
        train_log = train(train_loader, model, criterion, True, optimizer)
        model.eval()
        with torch.no_grad():
            val_log = train(val_loader, model, criterion, False, optimizer)

        print(
            'Training epoch [{}/{}], '
            'Training Loss:{:.4f}, '
            'Training IOU:{:.4f}, '
            'Training DICE:{:.4f}, '
            'Validation Loss:{:.4f}, '
            'Validation IOU:{:.4f},'
            'Validation Dice:{:.4f} '
                .format(
                epoch + 1, epochs, train_log['loss'], train_log['iou'], train_log['dice'],
                val_log['loss'], val_log['iou'], val_log['dice']))

        tmp = pd.Series([
            epoch,
            learning_rate,
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice']
        ], index=['epoch', 'lr', 'train_loss', 'train_iou',
                  'train_dice', 'val_loss', 'val_iou', 'val_dice'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('{}/model_outputs/{}/log.csv'.format(cur_path, file_name), index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            # if epoch > 30:
            torch.save(model.state_dict(), '{}/model_outputs/{}/model.pth'.format(cur_path, file_name))
            best_dice = val_log['dice']
            print("=>目前最佳 validation DICE:{:.4f}".format(best_dice))
            trigger = 0

        if 0 <= early_stopping <= trigger:
            print("=>因为dice陷入停滞，提前中止训练")
            break

        torch.cuda.empty_cache()

    model_result = pd.read_csv('{}/model_outputs/{}/log.csv'.format(cur_path, file_name))
    plot_metric(model_result, '{}/model_outputs/{}/'.format(cur_path, file_name), 'DICE')
    plot_metric(model_result, '{}/model_outputs/{}/'.format(cur_path, file_name), 'IOU', True)
    plot_loss(model_result, '{}/model_outputs/{}/'.format(cur_path, file_name), 'LOSS')
    view_output(model, file_name, dataset, shape, augmentations, cur_path)


if __name__ == '__main__':
    main()
