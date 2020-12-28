import torch
import random
import numpy as np
import HUMP_Functions as func
from PIL import Image
import tifffile
import pandas as pd
import HUMP_DATAGENERATOR
import matplotlib.pyplot as plt

##############IMPORTANT:    Path of Data
path = r"G:/HUMPChallange"


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
DataGenerator_TRAIN = HUMP_DATAGENERATOR.DATAGENERATOR(filenames=[X_train,y_train],
                                                       augmentation= func.get_training_augmentation(),
                                                       preprocessing = func.get_preprocessing(),
                                                       train_val_test_mode="train")
DataGenerator_Val =
n=10
fig = plt.figure()
for i in range(1,n):
    image,mask = DataGenerator_TRAIN[1]
    fig.add_subplot(1, n, i)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap="plasma")
plt.show()
