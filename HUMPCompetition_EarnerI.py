import torch
import random
import numpy as np
import HUMP_Functions as func
from PIL import Image
import tifffile
import pandas as pd
import HUMP_DATAGENERATOR_EarnerI as G
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pytorch_lightning as pl
import efficientunet as EUnet
from torchgeometry.losses import dice_loss
import torch

##############IMPORTANT:    Path of Data
path = r"G:/HUMPChallange"
BATCH_SIZE = 10


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
DataGenerator_TRAIN = G.DATAGENERATOR(filenames=[X_train,y_train],
                                                       augmentation= G.get_training_augmentation(),
                                                       preprocessing = G.get_preprocessing(),
                                                       train_val_test_mode="train")
DataGenerator_Val = G.DATAGENERATOR(filenames=[X_test,y_test],
                                                       augmentation= G.get_training_augmentation(),
                                                       preprocessing = G.get_preprocessing(),
                                                       train_val_test_mode="val")
train_loader = Data.DataLoader(DataGenerator_TRAIN, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=4, pin_memory=True)
val_loader = Data.DataLoader(DataGenerator_Val, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory= True)

model = EUnet.from_name("efficientnet-b5", n_classes=2, pretrained=False)
loss = dice_loss()
optimizer = torch.optim.Adam([dict(params=model.parameters(),lr=0.0001)])

trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
