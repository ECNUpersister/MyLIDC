import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import torch


def transform(albumentation, image, mask, shape):
    albu_transformations = albu.Compose([
        albu.ElasticTransform(alpha=1.1, alpha_affine=0.5, sigma=5, p=0.15),
        albu.HorizontalFlip(p=0.15),
        ToTensorV2()
    ])
    transformations = transforms.Compose([transforms.ToTensor()])
    if albumentation:
        image = image.reshape(shape, shape, 1)
        mask = mask.reshape(shape, shape, 1)
        mask = mask.astype('uint8')
        augmented = albu_transformations(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        mask = mask.reshape([1, shape, shape])
    else:
        image = transformations(image)
        mask = transformations(mask)

    image, mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)
    return image, mask
