import os
from glob import glob

import numpy as np

cur_path = 'G:/MyLIDC/data'
dataset_name = 'lidc_shape64'
mask_path_list = glob(os.path.join('{}/dataset/{}/test/Mask/*'.format(cur_path, dataset_name), "*.npy"))
for mask_path in mask_path_list:
    mask = np.load(mask_path)
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    if xmin == xmax or ymin == ymax:
        print(mask_path)
print(1)
