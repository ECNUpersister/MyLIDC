import os

import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
from sklearn.cluster import KMeans


def is_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def segment_lung(img):
    # function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    """
    This segments the Lung Image(Don't get confused with lung nodule lidc_segmentation)
    """
    imgk = img
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    middle = img[100:400, 100:400]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # remove the underflow bins
    img[img == max] = mean
    img[img == min] = mean

    # apply median filter
    img = median_filter(img, size=3)
    imgz = img
    # apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img = anisotropic_diffusion(img)

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    mask = np.ndarray([512, 512], dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([5, 5]))  # one last dilation
    # mask = morphology.erosion(mask, np.ones([20, 20]))
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    # return mask * img

    return imgz,mask,mask*imgk


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    img = plt.imread('C:/Users/15802/Desktop/0587_NI000_slice004.png')[:, :, 0]
    print(img.shape)
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].axis('off')
    a,b,c = segment_lung(img)
    plt.imsave("C:/Users/15802/Desktop/1.png", a, cmap=plt.cm.gray)
    plt.imsave("C:/Users/15802/Desktop/2.png", b, cmap=plt.cm.gray)
    plt.imsave("C:/Users/15802/Desktop/3.png", c, cmap=plt.cm.gray)
    ax[1].imshow(a, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[2].imshow(b, cmap=plt.cm.gray)
    ax[2].axis('off')
    ax[3].imshow(c, cmap=plt.cm.gray)
    ax[3].axis('off')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.show()
