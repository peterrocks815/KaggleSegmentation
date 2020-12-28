import efficientunet as EUnet
from torchgeometry.losses import dice_loss
import torch

model = EUnet.from_name("efficientnet-b5", n_classes=2, pretrained=False)
loss = dice_loss()
optimizer = torch.optim.Adam([dict(params=model.parameters(),lr=0.0001)])

def fitmodel(EPOCHS):
    for e in range(0,EPOCHS):
