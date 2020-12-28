import torch.utils.data as D
import cv2
import numpy as np
import albumentations as albu

class DATAGENERATOR(D.Dataset):
    def __init__(self,filenames, augmentation, preprocessing, train_val_test_mode):
        """train_val_mode: True=Training; False=Validation
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
        if self.train_val_test_mode is "train":
            img = self.func_read_filename(self.filenames[i][0])
            mask = cv2.imread(self.filenames[i][1], 0)
            mask = np.expand_dims(mask, axis=2)
            if self.augmentation:
                sample = self.augmentation(image=img, mask=mask)
                img,mask = sample["image"], sample["mask"]
            if self.preprocessing:
                sample = self.preprocessing(image=img, mask=mask)
                img,mask = sample["image"], sample["mask"]
            return img,mask
        if self.train_val_test_mode is "val":
            img = self.func_read_filename(self.filenames[i][0])
            mask = cv2.imread(self.filenames[i][1], 0)
            mask = np.expand_dims(mask, axis=2)
            return img,mask
        img = self.func_read_filename(self.filenames[i])
        return img

def get_training_augmentation():
    """
        To reduce the risk of overfitting,
        it is advisable to slightly modify the cropped images.

        """
    augmentation = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5)
    ]
    return albu.Compose(augmentation)

def get_preprocessing():
    process = [
        albu.Normalize(mean=(0.65459856, 0.48386562, 0.69428385),
                       std=(0.15167958, 0.23584107, 0.13146145),
                       max_pixel_value=255.0, always_apply=True, p=1.0),
    ]
    return albu.Compose(process)

