import os.path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation.transform import transform
import imageio


def view_output(model, dir, dataset, shape, albumentations, cur_path):
    index = 836
    model.load_state_dict(torch.load(os.path.join('{}/model_outputs/{}'.format(cur_path, dir), 'model.pth')))
    train_img = np.load('{}/dataset/{}/train/Image/LIDC-IDRI-0001/0001_NI000_slice000.npy'.format(cur_path, dataset))
    train_mask = np.load('{}/dataset/{}/train/Mask/LIDC-IDRI-0001/0001_MA000_slice000.npy'.format(cur_path, dataset))
    test_img = np.load(
        '{}/dataset/{}/test/Image/LIDC-IDRI-0{}/0{}_NI000_slice000.npy'.format(cur_path, dataset, index, index))
    test_mask = np.load(
        '{}/dataset/{}/test/Mask/LIDC-IDRI-0{}/0{}_MA000_slice000.npy'.format(cur_path, dataset, index, index))
    # Hu值转换大概在-1000到400之间，我们将其转换到0-255之间
    train_img = (train_img + 1000) / 1400 * 255
    train_img[train_img > 255] = 255
    train_img[train_img < 0] = 0
    train_img = train_img.astype(np.uint8)
    # Hu值转换大概在-1000到400之间，我们将其转换到0-255之间
    test_img = (test_img + 1000) / 1400 * 255
    test_img[test_img > 255] = 255
    test_img[test_img < 0] = 0
    test_img = test_img.astype(np.uint8)

    train_img, train_mask = transform(albumentations, train_img, train_mask, shape)
    test_img, test_mask = transform(albumentations, test_img, test_mask, shape)
    output_train_mask = model(train_img.unsqueeze(0).cuda())
    output_train_mask = torch.sigmoid(output_train_mask).data.cpu().numpy()[0][0]
    output_train_mask = output_train_mask > 0.5
    output_test_mask = model(test_img.unsqueeze(0).cuda())
    output_test_mask = torch.sigmoid(output_test_mask).data.cpu().numpy()[0][0]
    output_test_mask = output_test_mask > 0.5
    fig, ax = plt.subplots(2, 3)
    ax[0][0].imshow(train_img.numpy()[0], cmap=plt.cm.gray)
    ax[0][1].imshow(output_train_mask, cmap=plt.cm.gray)
    ax[0][2].imshow(train_mask.numpy()[0], cmap=plt.cm.gray)
    ax[1][0].imshow(test_img.numpy()[0], cmap=plt.cm.gray)
    ax[1][1].imshow(output_test_mask, cmap=plt.cm.gray)
    ax[1][2].imshow(test_mask.numpy()[0], cmap=plt.cm.gray)
    for i in range(len(ax)):
        for j in range(len(ax[0])):
            ax[i][j].axis('off')
    plt.show()
    # 保存样例图像
    dir_example = '{}/model_outputs/{}/example'.format(cur_path, dir)
    os.makedirs(dir_example, exist_ok=True)
    imageio.imsave(dir_example + '/训练图像.png', train_img.numpy()[0].astype(np.uint8))
    imageio.imsave(dir_example + '/训练输出.png', output_train_mask.astype(np.uint8))
    imageio.imsave(dir_example + '/训练标签.png', train_mask.numpy()[0].astype(np.uint8))
    imageio.imsave(dir_example + '/测试图像.png', test_img.numpy()[0].astype(np.uint8))
    imageio.imsave(dir_example + '/测试输出.png', output_test_mask.astype(np.uint8))
    imageio.imsave(dir_example + '/测试标签.png', test_mask.numpy()[0].astype(np.uint8))


if __name__ == '__main__':
    dir = '2021-09-09_18_21_37'
    dataset = 'lidc_shape64'
    shape = 64
    albumentations = True
    backbone = 'inceptionresnetv2'
    model = smp.UnetPlusPlus(
        encoder_name=backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        # decoder_attention_type='scse',
        decoder_use_batchnorm=True,
        # decoder_merge_policy='cat',
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model     channels (number of classes in your lidc_shape64)
    )
    view_output(model, dir, dataset, shape, albumentations)
