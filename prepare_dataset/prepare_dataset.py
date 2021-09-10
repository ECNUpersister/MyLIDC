import os
import warnings
from pathlib import Path
import numpy as np
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm

warnings.filterwarnings(action='ignore')
dataset = 'lidc_shape512'
shape = 512


# 如果最后用64*64需要滤除的文件：52、58、94、191、332、337、340、347、
# 415、463、487、576、624、655、701、703、709、829、834、997、1007
# 手动分割800以后的病人为验证集


class MakeDataSet:
    def __init__(self,
                 dicom_dir='E:/LIDC-IDRI',
                 image_dir='../{}/train/Image'.format(dataset),
                 mask_dir='../{}/train/Mask'.format(dataset),
                 mask_threshold=8,
                 confidence_level=0.5,
                 img_size=shape):
        # I found out that simply using os.listdir() includes the gitignore file
        self.IDRI_list = [f for f in os.listdir(dicom_dir) if not f.startswith('.')]
        self.IDRI_list.sort()
        self.img_path = image_dir
        self.mask_path = mask_dir
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.img_size = img_size

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)

        image_dir = Path(self.img_path)
        mask_dir = Path(self.mask_path)

        for patient in tqdm(self.IDRI_list):
            pid = patient  # LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            print("病人ID: {},Dicom尺寸: {},有 {} 个结节".format(pid, vol.shape, len(nodules_annotation)))

            if len(nodules_annotation) > 0:
                patient_image_dir = image_dir / pid
                patient_mask_dir = mask_dir / pid
                Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
                Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    if self.img_size == 512:
                        padding = [(512, 512), (512, 512), (0, 0)]
                    else:
                        _, bbox, _ = consensus(nodule, self.c_level, 0)
                        shape = vol[bbox].shape
                        pad1l = (self.img_size - shape[0]) // 2
                        pad1r = self.img_size - pad1l - shape[0]
                        pad2l = (self.img_size - shape[1]) // 2
                        pad2r = self.img_size - pad2l - shape[1]
                        padding = [(pad1l, pad1r), (pad2l, pad2r), (0, 0)]
                    mask, c_bbox, _ = consensus(nodule, self.c_level, padding)
                    bmats = np.array([a.bbox_matrix(pad=padding) for a in nodule])
                    imin, jmin, kmin = bmats[:, :, 0].min(axis=0)
                    imax, jmax, kmax = bmats[:, :, 1].max(axis=0)
                    if kmax - kmin >= 8:
                        lung_nodule_np_array = vol[c_bbox]
                        for nodule_slice in range(mask.shape[2]):
                            if np.sum(mask[:, :, nodule_slice]) <= self.mask_threshold:
                                continue
                            nodule_name = "{}_NI{}_slice{}".format(pid[-4:], prefix[nodule_idx], prefix[nodule_slice])
                            mask_name = "{}_MA{}_slice{}".format(pid[-4:], prefix[nodule_idx], prefix[nodule_slice])

                            np.save(patient_image_dir / nodule_name, lung_nodule_np_array[:, :, nodule_slice])
                            np.save(patient_mask_dir / mask_name, mask[:, :, nodule_slice])


if __name__ == '__main__':
    test = MakeDataSet()
    test.prepare_dataset()
