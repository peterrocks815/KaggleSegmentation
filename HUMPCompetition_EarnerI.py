import torch
import random
import numpy as np
import HUMP_Functions as func
from PIL import Image
import tifffile
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pytorch_lightning as pl
import HUMP_Model
from torchgeometry.losses import dice_loss
import torch
import Model_Tensorflow


##############IMPORTANT:    Path of Data
path = r"G:/HUMPChallange"
BATCH_SIZE = 10
EPOCHS = 100

#DEFINE SEED:   important to train always the same way
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#the pictures are to big, so we have to cut them and save smaller ones
#func.take_BIGIMG_and_save_RandomSmallImg_and_Mask(number_of_cuts=100,x_size=256,y_size=256, path=path)

#GET FILENAMES OF TRAINING AND VALIDATION DATA
X_train,y_train,X_test,y_test = func.get_filenames_of_TrainValData(path=path)
print("XTRAIN length: ", len(X_train))


EUNet = Model_Tensorflow.



DataGenerator_TRAIN = G.DATAGENERATOR(filenames=[X_train,y_train],
                                                               augmentation= G.get_training_augmentation(),
                                                               preprocessing = G.get_preprocessing(),
                                                               train_val_test_mode="train")
DataGenerator_Val = G.DATAGENERATOR(filenames=[X_test,y_test],
                                                               augmentation= G.get_training_augmentation(),
                                                               preprocessing = G.get_preprocessing(),
                                                               train_val_test_mode="val")