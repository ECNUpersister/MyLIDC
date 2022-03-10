import torch
# 检查过指标计算没问题

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    # we need to use sigmoid because the output of Unet is logit.
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


def dice_coef2(output, target):
    "This metric is for validation purpose"
    smooth = 1e-5

    output = output.view(-1)
    output = (output > 0.5).float().cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)

def dice_coef3(predicts,targets):
    smooth=1e-5
    predicts=[predict['masks'] for predict in predicts]
    targets=[target['masks'] for target in targets]
    for predict in predicts:
        predict=torch.cat(predict,dim=0)
        predict=torch.sum(predict, dim=0)

def dice_coeff(pred, target):
    smooth = 1e-5
    num = pred.shape[0]
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

