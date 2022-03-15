import os
from glob import glob

import numpy as np
import torch
predict_mask = torch.ones((2, 1, 4, 4))
m=predict_mask
predict_mask[0]=torch.zeros((1,4,4))
print(predict_mask)
print(m)