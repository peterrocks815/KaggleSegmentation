import segmentation_models_pytorch as smp
import HUMP_DATAGENERATOR_EarnerI as G
import efficientunet as EUnet
import torch.utils.data as Data


class model():
    def __init__(self, X_train,y_train, X_test,y_test):
        self.model = EUnet.from_name("efficientnet-b5", n_classes=2, pretrained=False)
        loss = smp.utils.losses.DiceLoss()
        metrics = [smp.utils.metrics.IoU(threshold=0.5)]
        optimizer = torch.optim.Adam([dict(params=self.model.parameters(),lr=0.0001)])


        DataGenerator_TRAIN = G.DATAGENERATOR(filenames=[X_train,y_train],
                                                               augmentation= G.get_training_augmentation(),
                                                               preprocessing = G.get_preprocessing(),
                                                               train_val_test_mode="train")
        DataGenerator_Val = G.DATAGENERATOR(filenames=[X_test,y_test],
                                                               augmentation= G.get_training_augmentation(),
                                                               preprocessing = G.get_preprocessing(),
                                                               train_val_test_mode="val")
        self.train_loader = Data.DataLoader(DataGenerator_TRAIN, batch_size=BATCH_SIZE,
                                       shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = Data.DataLoader(DataGenerator_Val, batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=4, pin_memory= True)

        self.train_epoch = smp.utils.train.TrainEpoch(
            self.model, loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device="device",
            verbose=True)

        self.val_epoch = smp.utils.train.ValidEpoch(
            self.model, loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device="device",
            verbose=True)

        best_loss = 1.0

    def fit(self,NUM_EPOCHS):
        train_losses, val_losses = [], []
        train_scores, val_scores = [], []

        for i in range(0, NUM_EPOCHS):

            print('\nEpoch: {}'.format(i))
            train_logs = self.train_epoch.run(self.train_loader)
            valid_logs = self.valid_epoch.run(self.valid_loader)

            train_losses.append(train_logs['dice_loss'])
            val_losses.append(valid_logs['dice_loss'])
            train_scores.append(train_logs['iou_score'])
            val_scores.append(valid_logs['iou_score'])

            if best_loss > valid_logs['dice_loss']:
                best_loss = valid_logs['dice_loss']
                torch.save(self.model, os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
                print('Model saved!')