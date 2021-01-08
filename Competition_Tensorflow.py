import SubCode.HUMP_Functions as func
import SubCode.DataGenerator_Tensorflow as tfG
import SubCode.Model_Tensorflow as model
from tensorflow.keras.optimizers import SGD, Adagrad, Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

##############IMPORTANT:    Path of Data
path = r"E:/HUMPChallange"
EPOCHS  = 100
NAME = "EU_NET-!HUMP!-{}".format(int(time.time()))


#the pictures are to big, so we have to cut them and save smaller ones
#func.take_BIGIMG_and_save_RandomSmallImg_and_Mask(number_of_cuts=100,x_size=256,y_size=256, path=path)

#GET FILENAMES OF TRAINING AND VALIDATION DATA
X_train,X_test,y_train,y_test = func.get_filenames_of_TrainValData(path=path)
print("XTRAIN length: ", len(X_train))
print("YTRAIN length: ", len(y_train))
print("XVAL length: ", len(X_test))
print("YVAL length: ", len(y_test))
print("HOLE length: ", len(X_train)+len(y_train)+len(X_test)+len(y_test))
DataGenerator_TRAIN = tfG.DATAGENERATOR(filenames=[X_train,y_train],
                                                       augmentation= tfG.get_training_augmentation(),
                                                       preprocessing = tfG.get_preprocessing(),
                                                       train_val_test_mode="train")
DataGenerator_Val = tfG.DATAGENERATOR(filenames=[X_test[0:10],y_test[0:10]],
                                                       augmentation= tfG.get_training_augmentation(),
                                                       preprocessing = tfG.get_preprocessing(),
                                                       train_val_test_mode="val")
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


print("INPUT_SHAPE: ", DataGenerator_TRAIN.__getitem__(0)[0].shape)
print("MASK/OUTPUT_SHAPE: ", DataGenerator_TRAIN.__getitem__(0)[1].shape)
print("LOG_Name: ", NAME)

EUNet = model.UEfficientNet(np.asarray(cv2.imread(X_train[0])).shape)
EUNet.compile(optimizer = SGD(lr=0.001), loss = model.dice_loss, metrics = ["acc"])
checkpoint_path = r"E:\HUMPChallange\model\eunet_weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
EUNet.load_weights(tf.train.latest_checkpoint(checkpoint_dir))


history = EUNet.fit_generator(generator= DataGenerator_TRAIN, validation_data=DataGenerator_Val, epochs=EPOCHS,
                              steps_per_epoch=DataGenerator_TRAIN.__len__(),
                              validation_steps=DataGenerator_Val.__len__(),
                              callbacks=[tensorboard],
                              workers=0)

print(np.argmax(EUNet.predict(DataGenerator_Val.__getitem__(0)[0])[0,:,:,0], axis=1))

rows,colums = 1,4
fig = plt.figure(figsize=(8,12))
fig1 = plt.subplot(rows,colums,1)
fig1.set_xticks([])
fig1.set_yticks([])
plt.imshow(DataGenerator_Val.__getitem__(0)[0][0,:,:,0], cmap="gray")
fig1 = plt.subplot(rows,colums,2)
fig1.set_xticks([])
fig1.set_yticks([])
plt.imshow(DataGenerator_Val.__getitem__(0)[1][0,:,:,1], cmap="gray")
fig1 = plt.subplot(rows,colums,3)
fig1.set_xticks([])
fig1.set_yticks([])
plt.imshow(EUNet.predict(DataGenerator_Val.__getitem__(0)[0])[0,:,:,0], cmap="gray")
fig1 = plt.subplot(rows,colums,4)
fig1.set_xticks([])
fig1.set_yticks([])
plt.imshow(EUNet.predict(DataGenerator_Val.__getitem__(0)[0])[0,:,:,1], cmap="gray")
plt.show()
EUNet.save(r"E:\HUMPChallange\model\model_unet.h5")
EUNet.save_weights(r"E:\HUMPChallange\model\eunet_weights.ckpt")
print(history)