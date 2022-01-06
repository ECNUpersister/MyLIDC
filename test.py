import numpy as np
import os
import matplotlib.pyplot as plt


pre_path = 'G:/MyLIDC/dataset/lidc_shape512/train/Image'
for dir in os.listdir(pre_path):
    print('当前已经处理到'+str(dir))
    dest_path = os.path.join('G:/MyLIDC/output/train/Image', dir)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for npy in os.listdir(os.path.join(pre_path, dir)):
        nparray = np.load(os.path.join(os.path.join(pre_path, dir), npy))
        png = npy.replace('npy', 'png')
        plt.imsave(os.path.join(dest_path, png), nparray, cmap='gray')
