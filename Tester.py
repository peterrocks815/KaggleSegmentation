import tensorflow as tf
from tensorflow.keras import Model
import SubCode.Model_Tensorflow as model
from tensorflow.keras.optimizers import SGD, Adagrad, Adam
import os
import SubCode.DataGenerator_Tensorflow as tfG
import matplotlib.pyplot as plt
import SubCode.HUMP_Functions as func

path = r"E:/HUMPChallange"

EUNet = model.UEfficientNet((512, 512, 3))
EUNet.compile(optimizer = SGD(lr=0.001), loss = model.dice_loss, metrics = ["acc"])
checkpoint_path = r"E:\HUMPChallange\model\eunet_weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
EUNet.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
X_train,X_test,y_train,y_test = func.get_filenames_of_TrainValData(path=path)
DataGenerator_Val = tfG.DATAGENERATOR(filenames=[X_test[0:100],y_test[0:100]],
                                                       augmentation= tfG.get_training_augmentation(),
                                                       preprocessing = tfG.get_preprocessing(),
                                                       train_val_test_mode="val")



fig = plt.figure(figsize=(8,12))
print(len(X_test))
rows,colums = 4, 8
for a in range(6):
    i = a*4
    print(i)
    fig1 = plt.subplot(rows, colums, i+1)
    fig1.set_xticks([])
    fig1.set_yticks([])
    plt.imshow(DataGenerator_Val.__getitem__(i)[0][0], cmap="gray")
    fig1 = plt.subplot(rows, colums, i + 2)
    fig1.set_xticks([])
    fig1.set_yticks([])
    plt.imshow(DataGenerator_Val.__getitem__(i)[1][0,:,:,0], cmap="gray")
    fig1 = plt.subplot(rows, colums, i + 3)
    fig1.set_xticks([])
    fig1.set_yticks([])
    #print(EUNet.predict(DataGenerator_Val.__getitem__(i)[0])[0].shape)
    plt.imshow(EUNet.predict(DataGenerator_Val.__getitem__(i)[0])[0,:,:,0], cmap="gray")
    fig1 = plt.subplot(rows, colums, i + 4)
    fig1.set_xticks([])
    fig1.set_yticks([])
    print(EUNet.predict(DataGenerator_Val.__getitem__(i)[0])[0].shape)
    plt.imshow(EUNet.predict(DataGenerator_Val.__getitem__(i)[0])[0, :, :, 1], cmap="gray")
plt.show()