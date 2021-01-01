from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import albumentations as albu

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
        return len(self.filenames)
    def __getitem__(self,i):
        if self.train_val_test_mode == "train":
            img = self.func_read_filename(self.filenames[i][0])
            mask = cv2.imread(self.filenames[i][1], 0)
            mask = np.expand_dims(mask, axis=2)
            if self.augmentation:
                sample = self.augmentation(image=img, mask=mask)
                img,mask = sample["image"], sample["mask"]
            if self.preprocessing:
                sample = self.preprocessing(image=img, mask=mask)
                img,mask = sample["image"], sample["mask"]
            return np.array([img], dtype = np.float32),np.array([mask], dtype=np.float32)
        if self.train_val_test_mode == "val":
            img = self.func_read_filename(self.filenames[i][0])
            mask = cv2.imread(self.filenames[i][1], 0)
            return np.array([img], dtype = np.float32),np.array([mask], dtype=np.float32)
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
        albu.IAAPerspective(p=0.5),
        albu.OneOf([                            #Use Just one Augmentation of:
            albu.RandomContrast(),
            albu.RandomGamma(),
            albu.RandomBrightness(),
            al.ColorJitter(brightness=0.07,
                           contrast=0.07,
                            saturation=0.1,
                           hue=0.1,
                           always_apply=False,p=0.3)],p=0.3),
        albu.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5)], p=0.0),
        albu.ShiftScaleRotate(),
    ]

    return albu.Compose(augmentation)

def get_preprocessing():
    process = [
        albu.Normalize(mean=(0.65459856, 0.48386562, 0.69428385),
                       std=(0.15167958, 0.23584107, 0.13146145),
                       max_pixel_value=255.0, always_apply=True, p=1.0),
    ]
    return albu.Compose(process)
