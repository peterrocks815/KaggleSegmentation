import torch
import random
import numpy as np
import HUMP_Functions as func
from PIL import Image
import tifffile
import pandas as pd
import HUMP_DATAGENERATOR_EarnerI as G
import matplotlib.pyplot as plt
import tensorflow as tf
import DataGenerator_Tensorflow as tfG
import Model_Tensorflow as model
import cv2
from tensorflow.keras.optimizers import SGD, Adagrad, Adam


##############IMPORTANT:    Path of Data
path = r"F:/HUMPChallange"
BATCH_SIZE = 10
EPOCHS  = 100


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
DataGenerator_TRAIN = tfG.DATAGENERATOR(filenames=[X_train,y_train],
                                                       augmentation= G.get_training_augmentation(),
                                                       preprocessing = G.get_preprocessing(),
                                                       train_val_test_mode="train")
DataGenerator_Val = tfG.DATAGENERATOR(filenames=[X_test,y_test],
                                                       augmentation= G.get_training_augmentation(),
                                                       preprocessing = G.get_preprocessing(),
                                                       train_val_test_mode="val")

EUNet = model.UEfficientNet(np.asarray(cv2.imread(X_train[0])).shape)
EUNet.compile(optimizer = SGD(lr=0.001), loss = model.dice_loss, metrics = ["acc"])
history = EUNet.fit_generator(generator= DataGenerator_TRAIN, validation_data=DataGenerator_Val, epochs=EPOCHS,
                              steps_per_epoch=DataGenerator_TRAIN.__len__(),
                              validation_steps=DataGenerator_Val.__len__(),
                              workers=0)
