#Maria Williams
#Last Modified: 6/15/22
#Classification training script
#not yet tested

#imports
import os
from google.colab import drive
from google.colab import files
#for image processing
!pip install pillow
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
#for metadata
import pandas as pd
#for data file structure
!pip install split-folders
import splitfolders
import shutil
#for modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Subset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchsummary import summary
#for metrics
from sklearn.metrics import mean_squared_error

#set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def LoaderPrep():
    #pull up image metadata
    filein = '/content/gdrive/My Drive/School/cn_age_df.csv'
    data = pd.read_csv('filein')

    #get images
    !unzip '/content/gdrive/My Drive/School/image_data.zip' > /dev/null

    #establish the file structure required for the dataloaders
    folders = ['50','60','70','80','90']
    os.mkdir("/content/data/")

    for i in folders:
    base = '/content/data'
    path = os.path.join(base, i,)
    os.mkdir(path)

    #this is the slice picked from the 3D image
    imslice = 85
    
    #for each image: pull out image slice, convert to jpeg, then move to class (age) folder
    #note: conversion to jpeg lets us use the datasets.Imageloader
    #note: images will be converted to tensors during transform since loaders can't process images
    for i in range(len(data)):
        #first load the image, pull out a slice and convert it to jpeg
        imgpl1 = "/content/image_data/" + data.loc[i][0] + '.npy'
        imgjpl = "/content/image_data/" + data.loc[i][0] + '.jpg'
        image3D = np.load(imgpl1)
        im = Image.fromarray((image3D[imslice] * 255).astype(np.uint8))
        im.save(imgjpl)

        #then loop through to determine which folder it belongs to
        for f in folders:
            if data.loc[i][5] < int(f):
            #if less than the folder name, move to the folder before f
            imgpl2 = "/content/data/" + str(int(f)-10) + "/" + data.loc[i][0] + '.jpg'
            os.rename(imgjpl, imgpl2)
            break
            elif data.loc[i][5] >= 90:
            imgpl2 = "/content/data/90/" + data.loc[i][0] + '.jpg'
            os.rename(imgjpl, imgpl2)
            break

    #since there are only 7 in the 50-60 decade, discard that folder and update folders list
    shutil.rmtree("/content/data/50")
    folders = ['60','70','80','90']

    #training/test split ratio: 90/10
    splitfolders.ratio('/content/data', output='/content/data', ratio=(.9, 0.1))

    #get datasets
    Tdataset = datasets.ImageFolder('/content/data/train/')
    Vdataset = datasets.ImageFolder('/content/data/val/')

    #define transformation - same for both train and val
    data_transform = transforms.Compose([ transforms.Resize((224, 224)),
                                            transforms.Grayscale(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5],std=[0.5])])

    #get datasets with transformation
    Tdataset = datasets.ImageFolder('/content/data/train/', transform=data_transform)
    Vdataset = datasets.ImageFolder('/content/data/val/', transform=data_transform)

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

#prepare the pre-trained model
def MakeModel():
    # Load a resnet18 pre-trained model
    model_resnet = torchvision.models.resnet18(pretrained=True)

    # Shut off autograd for all layers to freeze model so the layer weights are not trained
    for param in model_resnet.parameters():
        param.requires_grad = False
        
    # Replace the resnet input layer to take in grayscale images (1 input channel), since it was originally trained on color (3 input channels)
    in_channels = 1
    model_resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the resnet final layer with a new fully connected Linear layer we will train on our task
    # Number of out units is number of classes (5)
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs, 5)

    return model_resnet
    
#train function
def train_model(model,criterion,optimizer,loader,n_epochs,device):
    
    loss_over_time = [] # to track the loss as the network trains
    
    model = model.to(device) # Send model to GPU if available
    model.train() # Set the model to training mode
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        running_corrects = 0
        
        for i, data in enumerate(loader):
            # Get the input images and labels, and send to GPU if available
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the weight gradients
            optimizer.zero_grad()

            # Forward pass to get outputs
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backpropagation to get the gradients with respect to each weight
            loss.backward()

            # Update the weights
            optimizer.step()

            # Convert loss into a scalar and add it to running_loss
            running_loss += loss.item()

            # Convert loss into a scalar and add it to running_loss
            running_loss += loss.item() * inputs.size(0)
            # Track number of correct predictions
            running_corrects += torch.sum(preds == labels.data)
            
        # Calculate and display average loss and accuracy for the epoch
        epoch_loss = running_loss / len(Tdataset)
        epoch_acc = running_corrects.double() / len(Tdataset)
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        loss_over_time.append(epoch_loss)

    return loss_over_time


# Main
Train_loader, Val_loader = LoaderPrep()
model = MakeModel()
n_epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cost_path = train_model(model,criterion,optimizer,Train_loader,n_epochs,device)

# Save the entire model
torch.save(model, "/content/model.pt")