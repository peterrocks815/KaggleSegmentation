from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
import tifffile
import json
import numpy as np
import urllib3
import base64
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

######################GET IMAGES
def subfunc_ShowIMGandMask(CUT_IMG,CUT_MASK):
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(CUT_IMG)
    ax[1].imshow(CUT_MASK)
    plt.show()

def subfunc_CalcCutPos(X_size,x_size,Y_size,y_size):
    m = np.random.randint(0, X_size - x_size)
    n = np.random.randint(0, Y_size - y_size)
    o = m + x_size
    p = n + y_size
    return m,n,o,p

def subfunc_GETIMG(m,n,o,p,row,i,IMG,MASK,path):
    CUT_PATH_IMG = path + "/CutTrain/" + row["id"] + "_%05d_img.png" % i
    CUT_PATH_MASK = path + "/CutTrain/" + row["id"] + "_%05d_mask.png" % i
    CUT_IMG = Image.fromarray(IMG[m:o, n:p])
    CUT_MASK = Image.fromarray(MASK[m:o, n:p])
    return CUT_IMG,CUT_MASK, CUT_PATH_IMG, CUT_PATH_MASK

def subfunc_SaveMASK_Switcher(X_size,x_size,Y_size,y_size,row,i,IMG,MASK,path, MODUS=0):
    if MODUS == 0:
        treshold = lambda maskinput: maskinput>3000
        modus = "Yuhu! Mask :D  "
    else:
        treshold = lambda maskinput: maskinput<=3000
        modus = "NO Mask :(  "

    finder = True
    counter = 0
    while finder:
        m, n, o, p = subfunc_CalcCutPos(X_size,x_size,Y_size,y_size)
        CUT_IMG,CUT_MASK,CUT_PATH_IMG,CUT_PATH_MASK = subfunc_GETIMG(m,n,o,p, row, i, IMG,MASK,path)
        maskinput = np.sum(np.asarray(CUT_MASK))
        if treshold(maskinput):
            print("Modus: {}; CutPos: [{}:{},{}:{}];  COUNTER: {}; Maskinput: {}; Name: {}".format(modus, m, n, o, p,
                                                                                           counter,
                                                                                           maskinput,CUT_PATH_IMG))
            CUT_IMG.save(CUT_PATH_IMG, "png")
            CUT_MASK.save(CUT_PATH_MASK, "png")
            #subfunc_ShowIMGandMask(CUT_IMG, CUT_MASK)
            finder = False
        counter = counter+1

def take_BIGIMG_and_save_RandomSmallImg_and_Mask(number_of_cuts=10,x_size=512,y_size=512, path=""):
    """Takes the big images and saves small images randomly cut out of the bit img.
        Number_of_cuts: tells how many small images will saved,
        x,y_size: Size of small imm,
        trainlist,masklist: A list containing the training/masks paths,
        path: the path of the file the small images will be saved in"""

    print("Padas open: ",path+"/train.csv")
    train_df = pd.read_csv(path+"/train.csv")

    for index,row in train_df.iterrows():
        print("ImgPath: ",path +"/train/"+ row["id"] +".tiff")
        IMG = tifffile.imread(path +"/train/"+ row["id"] +".tiff")
        X_size, Y_size = IMG.shape[0], IMG.shape[1]
        print("OriginalImgSize: ", (X_size, Y_size))
        MASK = rle2mask(row["encoding"], (Y_size, X_size))
        for i in range(number_of_cuts):
            subfunc_SaveMASK_Switcher(X_size, x_size, Y_size, y_size, row, i, IMG, MASK,path, MODUS = 0)
            subfunc_SaveMASK_Switcher(X_size, x_size, Y_size, y_size, row, i+number_of_cuts, IMG, MASK,path, MODUS = 1)

def get_filenames_of_TrainValData(path):
    """Takes the Path where all the Data is stored.
    Goes into the created TrainingFile: "CutTrain", load all files and returns the pathnames
    splited into train_img,train_mask, val_img, val_mask """

    train_data = sorted(glob.glob(os.path.join(path + "/CutTrain/*_img.png")))
    mask_data = sorted(glob.glob(os.path.join(path+"/CutTrain/*_mask.png")))
    return train_test_split(train_data[0:100],mask_data[0:100])

def get_filenames_of_TestData(path):
    return glob.glob(os.path.join(path,"*.tiff"))
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
def rle2mask(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T
def gridCut_of_BigImg(img,x_size,y_size):
    X_size,Y_size = img.shape
    images = []
    for x in range(X_size/x_size):
        for y in range(Y_size/y_size):
            images.append(img[x:x+x_size,y:y+y_size])
    return images
def showImgAndMask(img,mask):
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    plt.show()



def prepocessMask(mask,w1,w2):
    mask = to_categorical(mask, num_classes=2)
    mask[:,:,0] = mask[:,:,0]*w1
    mask[:, :, 1] = mask[:, :, 0] * w2
    return np.array([mask])