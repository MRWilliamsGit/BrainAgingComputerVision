#Maria Williams
#Last Modified: 6/15/22
#Regression training script

#imports
import os
import shutil

#for data processing
import scripts.loaders
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

#for modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import mean_squared_error

#train function
def train_model(model,criterion,optimizer,loader,n_epochs,device):
    
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

def MakeAndTrain(Train_loader, device):
    
    #make loaders, model, optimizer
    model = MakeModel()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    #train model
    cost_path = train_model(model = model,
                            criterion = nn.MSELoss(),
                            optimizer = optimizer,
                            loader = Train_loader,
                            n_epochs = 25,
                            device = device)
    
    # Save the entire model
    #torch.save(model, "/home/ec2-user/environment/BrainAgingComputerVision/models/model2.pt")
    return model