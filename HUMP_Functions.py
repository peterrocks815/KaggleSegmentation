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

def take_BIGIMG_and_save_RandomSmallImg_and_Mask(number_of_cuts=10,x_size=256,y_size=256, path=""):
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
        print(X_size, Y_size)
        MASK = rle2mask(row["encoding"], (Y_size, X_size))
        for i in range(number_of_cuts):
            m = np.random.randint(0, X_size - x_size)
            n = np.random.randint(0, Y_size - y_size)
            o = m + x_size
            p = n + y_size
            CUT_PATH_IMG = path + "/CutTrain/" + row["id"] + "_%03d_img.png" % i
            CUT_PATH_MASK = path + "/CutTrain/" + row["id"] + "_%03d_mask.png" % i
            print("CUTPostion: ", "[{}:{},{}:{}]".format(m,n,o,p))
            CUT_IMG = Image.fromarray(IMG[m:o,n:p])
            CUT_MASK = Image.fromarray(MASK[m:o,n:p])
            CUT_IMG.save(CUT_PATH_IMG,"png")
            CUT_MASK.save(CUT_PATH_MASK,"png")







def get_filenames_of_testImg(path):
    return glob.glob(os.path.join(path,"*.tiff"))
def showImage_List_path(path):
    fig = plt.figure()
    for i in range(1,len(path)+1):
        img = Image.open(path[i])
        fig.add_subplot(1,len(path),i)
        plt.imshow(img)
    plt.show()
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