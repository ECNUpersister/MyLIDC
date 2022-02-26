import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)

        return 1 - dice.sum() / num

def dice_cal(images,targets):
    dice=0
    smooth = 1e-5
    for image,target in images,targets:
        image=image[1,:,:]
        image=image.view(1,-1)
        target=target.view(1,-1)
        intersection = (image * target)
        dice += (2. * intersection.sum(1) + smooth) / (image.sum(1) + target.sum(1) + smooth)
    return dice/len(images)
