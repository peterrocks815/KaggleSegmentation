from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import albumentations as albu
import SubCode.HUMP_Functions as func
import matplotlib.pyplot as plt


class DATAGENERATOR(Sequence):
    def __init__(self,filenames, augmentation, preprocessing, train_val_test_mode):
        """Generates the Input for the NN.
            filenames: train,val: shape(N,2) = [train_array,label_array]
                        test: shape(N) = test_array
                        *_array: Array of all filenames
            augmentation: function with augmentation
            preprocessing: function with preprocessing
            train_val_test_mode: "train"=training batch, "val"=Validation batch, "test"= Testbatch
        """
        self.train_val_test_mode = train_val_test_mode
        self.filenames = filenames
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    def func_read_filename(self,file):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    def __len__(self):
        return len(self.filenames[0])
    def __getitem__(self,i):
        if self.train_val_test_mode == "train":
            img = self.func_read_filename(self.filenames[0][i])
            mask = cv2.imread(self.filenames[1][i], 0)
            mask = np.expand_dims(mask, axis=2)
            if self.augmentation:
                sample = self.augmentation(image=img, mask=mask)
                img,mask = sample["image"], sample["mask"]
            if self.preprocessing:
                sample = self.preprocessing(image=img, mask=mask)
                img,mask = sample["image"], sample["mask"]
            img =   np.array([img])
            mask = func.prepocessMask(mask,w1=0.5,w2=5)
            #func.showImgAndMask(img[0], mask[0])
            return img,mask

        if self.train_val_test_mode == "val":
            img = self.func_read_filename(self.filenames[0][i])
            mask = cv2.imread(self.filenames[1][i], 0)
            img = np.array([img])
            mask = func.prepocessMask(mask,w1=0.5,w2=5)
            #func.showImgAndMask(img[0], mask[0])
            return img,mask

        img = self.func_read_filename(self.filenames[i])
        return np.array([img], dtype = np.float32)

def get_training_augmentation():
    """
        To reduce the risk of overfitting,
        it is advisable to slightly modify the cropped images.

        """
    augmentation = [
        albu.HorizontalFlip(p=0.5),             #50% of horizontal Flip
        albu.VerticalFlip(p=0.5),               #50% of vertical Flip
        albu.RandomRotate90(p=0.5),             #50% of Roation of +-90°
        albu.IAAAdditiveGaussianNoise(p=0.2),   #Implement 20% Noise
        #albu.IAAPerspective(p=0.5),

    ]

    return albu.Compose(augmentation)

def get_preprocessing():
    process = [
        albu.Normalize(mean=(0.65459856, 0.48386562, 0.69428385),
                       std=(0.15167958, 0.23584107, 0.13146145),
                       max_pixel_value=255.0, always_apply=True, p=1.0),
    ]
    return albu.Compose(process)
