import pytorch_lightning as pl
import torch
from PIL import Image
import os
import numpy as np

class DATAGENERATOR(pl.LightningModule):
    def __init__(self, train_val_mode):
        """train_val_mode: True=Training; False=Validation
        """
        self.train_val_mode = train_val_mode
        train_list = []
        label_list = []
        for i in os.listdir("./CutTrain"):
            if "img" in i:
                img = Image.open("./CutTrain/" + i)
                train_list.append(np.array(img))
            else:
                mask = Image.open("./CutTrain/" + i)
                label_list.append(np.array(mask))
        train_data = torch.Tensor(train_list)
        labels = torch.Tensor(label_list)
        print(labels.size())

        self.X_train = train_data[:320]
        self.y_train = labels[:320]
        self.X_val = train_data[320:]
        self.y_val = labels[320:]
        self.num_samples_train = self.X_train.size()[0]
        self.num_samples_val = self.X_val.size()[0]

    def __len__(self):
        if self.train_val_mode:
            return self.num_samples_train
        else:
            return self.num_samples_val

    def __getitem__(self, idx):
        if self.train_val_mode:
            return self.X_train[idx], self.y_train[idx]
        else:
            return self.X_val[idx], self.y_val[idx]


train_datagenerator = DATAGENERATOR(True)
val_datagenerator = DATAGENERATOR(False)
#train_datagenerator is a 320 256 256 sized tensor
#val_datagenerator is a 80 256 256 sized tensor

