#Maria Williams
#Last Modified: 6/15/22
#These libraries create loaders for different models

#imports
import os
#for image processing
import zipfile
import random
import numpy as np
import tensorflow as tf
from PIL import Image
#from matplotlib import pyplot as plt
#for metadata
import pandas as pd
#for data file structure
import splitfolders
import shutil

#for modeling
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Subset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchsummary import summary
#for metrics
#from sklearn.metrics import mean_squared_error

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
    imgpath = self.root + self.data_paths[idx]
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

#must pass path to folder that contains image_data.zip and cn_age_df.csv
#must pass the slice number you want (default 85)
def ClassLoaders(where_stuff, imslice=85):
    #get image metadata
    csvin = os.path.join(where_stuff, 'cn_age_df.csv')
    data = pd.read_csv(csvin)

    #get images - will unzip in same folder
    imin= os.path.join(where_stuff, 'image_data.zip')
    imout= os.path.join(where_stuff, 'image_data')
    with zipfile.ZipFile(imin, 'r') as zipObj:
        zipObj.extractall(imout)
        
    #establish the file structure required for the dataloaders
    folders = ['50','60','70','80','90']
    dfolder = os.path.join(where_stuff, 'data')
    os.mkdir(dfolder)
    for i in folders:
        path = os.path.join(dfolder, i,)
        os.mkdir(path)

    #for each image: pull out image slice, convert to jpeg, then move to class (age) folder
    #note: conversion to jpeg lets us use the datasets.Imageloader
    #note: images will be converted to tensors during transform since loaders can't process images
    flist = os.listdir(imout)
    for m in flist:
        im = m.split(".")[0]
        imgpl1 = imout+"/"+im+'.npy'
        imgjpl = imout+"/"+im+'.jpg'
    
    #full data version
    #for i in range(len(data)):
        #first load the image, pull out a slice and convert it to jpeg
        #imgpl1 = imout + "/" + data.loc[i][0] + '.npy'
        #imgjpl = imout + "/" + data.loc[i][0] + '.jpg'
        image3D = np.load(imgpl1)
        this = Image.fromarray((image3D[imslice] * 255).astype(np.uint8))
        this.save(imgjpl)

        #then loop through to determine which folder it belongs to
        age = data.loc(data[0]==im)[5]
        for f in folders:
            if age < int(f):
                #if less than the folder name, move to the folder before f
                imgpl2 = dfolder + "/" + str(int(f)-10) + "/" + im + '.jpg'
                os.rename(imgjpl, imgpl2)
                break
            elif age >= 90:
                imgpl2 = dfolder + "/90/" + im + '.jpg'
                os.rename(imgjpl, imgpl2)
                break

    #since there are only 7 in the 50-60 decade, discard that folder and update folders list
    shutil.rmtree(dfolder+"/50")
    folders = ['60','70','80','90']

    #training/test split ratio: 90/10
    splitfolders.ratio(dfolder, output=dfolder, ratio=(.9, 0.1))

    #get datasets
    Tdataset = datasets.ImageFolder(dfolder+'/train/')
    Vdataset = datasets.ImageFolder(dfolder+'/val/')

    #define transformation - same for both train and val
    data_transform = transforms.Compose([ transforms.Resize((224, 224)),
                                            transforms.Grayscale(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5],std=[0.5])])

    #get datasets with transformation
    Tdataset = datasets.ImageFolder(dfolder+'/train/', transform=data_transform)
    Vdataset = datasets.ImageFolder(dfolder+'/val/', transform=data_transform)

    #add to loaders (batch size of 36 would be square root of dataset)
    batch_size = 30
    Train_loader = torch.utils.data.DataLoader(Tdataset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    Val_loader = torch.utils.data.DataLoader(Vdataset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

    #set up class names for visualizations and checking
    idx_to_class = {i:j for i,j in enumerate(folders)}
    class_to_idx = {v:k for k,v in idx_to_class.items()}

    return Train_loader, Val_loader
    
#make sure work
#print(os.getcwd())
T, V = ClassLoaders('/home/ec2-user/environment/BrainAgingComputerVision/data')
