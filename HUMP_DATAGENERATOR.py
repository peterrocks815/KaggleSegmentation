import tensorflow as tf
from tensorflow.keras.utils import Sequence


class DATAGENERATOR(Sequence):
    def __init__(self, train_val_mode):
        """train_val_mode: True=Training; False=Validation
        """
        self.train_val_mode = train_val_mode

    def __len__(self):

    def __getitem__(self):
