#Maria Williams
#Last Modified: 6/15/22
#Regression training script
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

#define class for the images
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


def LoaderPrep():
    #pull up image metadata
    filein = '/content/gdrive/My Drive/School/cn_age_df.csv'
    data = pd.read_csv('filein')

    #get images
    !unzip '/content/gdrive/My Drive/School/image_data.zip' > /dev/null

    #divide files into training and test sets, ratio: 90/10
    #note this does not ensure that classes are equivalently represented
    flist = os.listdir('/content/image_data/')
    random.shuffle(flist)
    tlen = round(len(flist)*.9)
    tlist = flist[0:tlen]
    vlist = flist[tlen:len(flist)]
    os.mkdir('/content/image_data/train/')
    os.mkdir('/content/image_data/val/')

    for f in tlist:
    imgpl1 = "/content/image_data/" + f
    imgpl2 = "/content/image_data/train/" + f
    os.rename(imgpl1, imgpl2)
    for f in vlist:
    imgpl1 = "/content/image_data/" + f
    imgpl2 = "/content/image_data/val/" + f
    os.rename(imgpl1, imgpl2)

    #define transformation - same for both train and val
    data_transform = transforms.Compose([ transforms.Resize((224, 224)),
                                         transforms.Grayscale(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5],std=[0.5])])

    #get data
    train_data = BrainBits(img_path='/content/image_data/train/', label_file=data, transform=data_transform)
    test_data = BrainBits(img_path='/content/image_data/val/', label_file=data, transform=data_transform)

    #make loaders
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

    return train_loader, test_loader
    
#train function
def train_model2(model,criterion,optimizer,loader,n_epochs,device):
    
    loss_over_time = [] # to track the loss as the network trains
    model = model.to(device) # Send model to GPU if available
    model.train() # Set the model to training mode
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        
        for i, data in enumerate(loader):
            
            # Get the input images and labels, and send to GPU if available
            inputs, labels = data[0].to(device), data[1].to(device)
            labels.resize_(len(labels), 1)

            # Zero the weight gradients
            optimizer.zero_grad()

            # Forward pass to get outputs
            outputs = model.forward(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels.float())

            # Backpropagation to get the gradients with respect to each weight
            loss.backward()

            # Update the weights
            optimizer.step()
            
            # Collect loss
            running_loss += loss.item()
            
        # Calculate and display average loss for the epoch
        epoch_loss = running_loss / len(loader)
        print('Loss: {:4f}'.format(epoch_loss))

        loss_over_time.append(epoch_loss)

    return loss_over_time

def MakeModel():
    # Load a resnet18 pre-trained model
    regmodel = torchvision.models.resnet18(pretrained=True)

    # Shut off autograd for all layers to freeze model so the layer weights are not trained
    for param in regmodel.parameters():
        param.requires_grad = False
        
    # Replace the resnet input layer to take in grayscale images (1 input channel), since it was originally trained on color (3 input channels)
    in_channels = 1
    regmodel.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the resnet final layer with a new fully connected Linear layer we will train on our task
    # Number of out units is 1 because regression
    num_ftrs = regmodel.fc.in_features
    regmodel.fc = nn.Linear(num_ftrs, 1)

    return regmodel


# Main
Train_loader, Val_loader = LoaderPrep()
model = MakeModel()
n_epochs = 25
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cost_path = train_model2(model,criterion,optimizer,Train_loader,n_epochs,device)

# Save the entire model
torch.save(model, "/content/model.pt")