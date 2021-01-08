import tensorflow as tf
from tensorflow.keras import Model
import SubCode.Model_Tensorflow as model
from tensorflow.keras.optimizers import SGD, Adagrad, Adam
import os
import SubCode.DataGenerator_Tensorflow as tfG
import matplotlib.pyplot as plt

EUNet = model.UEfficientNet((512, 512, 3))
print(EUNet.summary())
EUNet.compile(optimizer = SGD(lr=0.001), loss = model.dice_loss, metrics = ["acc"])
checkpoint_path = r"E:\HUMPChallange\model\eunet_weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
EUNet.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

layer_list = [481,475,453,350,430]
a= 0,427,421,418,415,472, 466,463,460,441,438,
#for i in range(20):
#    a = 427-i
#    print(a, EUNet.layers[a].name)

outputs = [EUNet.layers[i].output for i in layer_list]
model_short = Model(inputs=EUNet.inputs, outputs=outputs)

X_test = [r"E:\HUMPChallange\CutTrain\2f6ecfcdf_00949_img.png"]
y_test = [r"E:\HUMPChallange\CutTrain\2f6ecfcdf_00949_mask.png"]
DataGenerator_Val = tfG.DATAGENERATOR(filenames=[X_test,y_test],
                                                       augmentation= tfG.get_training_augmentation(),
                                                       preprocessing = tfG.get_preprocessing(),
                                                       train_val_test_mode="val")
tester = model_short.predict(DataGenerator_Val.__getitem__(0)[0])
print(DataGenerator_Val.__getitem__(0)[1].shape)
print(len(tester))



rows,colums = 20,8
fig = plt.figure(figsize=(8,12))
fig1 = plt.subplot(rows,colums,1)
fig1.set_xticks([])
fig1.set_yticks([])
plt.imshow(DataGenerator_Val.__getitem__(0)[0][0], cmap="gray")
fig1 = plt.subplot(rows,colums,2)
fig1.set_xticks([])
fig1.set_yticks([])
plt.imshow(DataGenerator_Val.__getitem__(0)[1][0], cmap="gray")
fig1 = plt.subplot(rows,colums,3)
fig1.set_xticks([])
fig1.set_yticks([])
plt.imshow(EUNet.predict(DataGenerator_Val.__getitem__(0)[0])[0], cmap="gray")

for n,ftr in enumerate(tester[1:]):
    print(EUNet.layers[layer_list[n]].name ,ftr.shape)
    if int(ftr.shape[-1]/4) > 8:
        a = 8
    else:
        a = int(ftr.shape[-1]/4)
    for i in range(1,9):
        fig1 = plt.subplot(rows,colums,8+i+n*8)
        fig1.set_xticks([])
        fig1.set_yticks([])
        plt.imshow(ftr[0,:,:,i-1], cmap="gray")
plt.show()


