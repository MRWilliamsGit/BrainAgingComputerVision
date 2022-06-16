#Maria Williams
#Last Modified: 6/15/22

#imports
import os
import shutil

#for data processing
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import zipfile
from PIL import Image

#for modeling
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Subset, DataLoader, Dataset

#define class for the regression images
class BrainBits(Dataset):
  #receives the folder of images, the csv file of data, and the transform
  def __init__(self, img_path, label_file, transform):
    self.root = img_path
    self.data_paths = [f for f in sorted(os.listdir(img_path))]
    self.dataf = label_file
    self.transform = transform

  def __getitem__(self, idx):

    #get file then set 85 slice as image
    imgpath = self.root + "/" + self.data_paths[idx]
    image3D = np.load(imgpath)
    img = Image.fromarray((image3D[85] * 255).astype(np.uint8))

    #get name of file
    imgn = self.data_paths[idx].split(".")
    imgn = imgn[0]

    #find filename in dataf, set age as label
    line = self.dataf.loc[self.dataf['Data ID'] == imgn]
    label = line.iloc[0][5]

    #apply transform
    if self.transform:
      img = self.transform(img)
    return img, label

  def __len__(self):
    return len(self.data_paths)

#must pass path to folder that contains image_data.zip and cn_age_df.csv
def RegressionLoaders(where_stuff):
    #get image metadata
    csvin = os.path.join(where_stuff, 'cn_age_df.csv')
    data = pd.read_csv(csvin)

    #get images - will unzip in same folder
    imin= os.path.join(where_stuff, 'image_data.zip')
    imout= os.path.join(where_stuff, 'image_data')
    with zipfile.ZipFile(imin, 'r') as zipObj:
        zipObj.extractall(imout)

    #divide files into training and test sets, ratio: 90/10
    #note: this does not ensure that classes are equivalently represented

    #step 1: make a list of the image files
    flist = os.listdir(imout)
    #step 2: shuffle
    random.shuffle(flist)
    #step 3: make two lists of train and val images
    tlen = round(len(flist)*.9)
    tlist = flist[0:tlen]
    vlist = flist[tlen:len(flist)]
    #step 4: make directories
    os.mkdir(os.path.join(where_stuff, 'train'))
    os.mkdir(os.path.join(where_stuff, 'val'))

    #step 5: sort the images as train and val
    for f in tlist:
        imgpl1 = os.path.join(where_stuff, 'image_data', f)
        imgpl2 = os.path.join(where_stuff, 'train', f)
        os.rename(imgpl1, imgpl2)
    for f in vlist:
        imgpl1 = os.path.join(where_stuff, 'image_data', f)
        imgpl2 = os.path.join(where_stuff, 'val', f)
        os.rename(imgpl1, imgpl2)

    #define transformation - same for both train and val
    data_transform = transforms.Compose([ transforms.Resize((224, 224)),
                                         transforms.Grayscale(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5],std=[0.5])])

    #get data
    train_data = BrainBits(img_path=os.path.join(where_stuff, 'train'), label_file=data, transform=data_transform)
    test_data = BrainBits(img_path=os.path.join(where_stuff, 'val'), label_file=data, transform=data_transform)

    #make loaders
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

    return train_loader, test_loader