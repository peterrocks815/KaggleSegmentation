import torch
import random
import numpy as np
import HUMP_Functions as func
from PIL import Image
import tifffile
import pandas as pd
#DEFINE SEED:   important to train always the same way

SEED = 12


np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DATA_TRAIN_DIR= r"G:\HUMPChallange\test"
DATA_TEST_DIR = r""

path = r"G:/HUMPChallange"
func.take_BIGIMG_and_save_RandomSmallImg_and_Mask(number_of_cuts=100,x_size=256,y_size=256, path=path)

