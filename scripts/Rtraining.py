#Maria Williams
#Last Modified: 6/16/22
#Regression training script

#imports
import os
import shutil

#for data processing
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

#validation function for early stopping
def validation(model, device, loader, loss_function):

    model.eval()
    loss_total = 0
    
    with torch.no_grad():
        # Get a batch of validation images
        images, labels = iter(loader).next()
        images, labels = images.to(device), labels.to(device)
        # Get predictions
        preds = model(images)
        
        loss = loss_function(preds, labels)
        loss_total += loss.item()

    return loss_total / len(loader)

#train function
#includes early stopping: if validation loss increases twice in a row, training will stop
def train_model(model,criterion,optimizer,loader,vloader,n_epochs,device):
    
    # Early stopping
    last_loss = 100
    patience = 2
    trigger_times = 0
    
    #prep
    loss_over_time = [] # to track the loss as the network trains
    V_loss_over_time = [] # to track the loss as the network trains
    model = model.to(device) # Send model to GPU if available
    model.train() # Set the model to training mode
    
    #train
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
            
        # Calculate and display the losses for each epoch
        epoch_loss = running_loss / len(loader)
        current_loss = validation(model, device, vloader, criterion)
        print('Training Loss: {:4f}, Validation Loss: {:4f}'.format(epoch_loss, current_loss))

        
        #track
    
        if current_loss > last_loss:
            trigger_times += 1
            print('Loss Increase Count:', trigger_times)
    
            if trigger_times >= patience:
                print('Loss Increase Count:', trigger_times)
                print('Early stopping!')
                return model
    
        else:
            #print('trigger times: 0')
            trigger_times = 0
        
        last_loss = current_loss
        loss_over_time.append(epoch_loss)
        V_loss_over_time.append(current_loss)

    return loss_over_time, V_loss_over_time

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

#runs everything, optional variable for if want to save model
def MakeAndTrain(Train_loader, Val_loader, device, where=""):
    
    #make loaders, model, optimizer
    model = MakeModel()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    #train model
    T_cost_path, V_cost_path = train_model(model = model,
                                criterion = nn.MSELoss(),
                                optimizer = optimizer,
                                loader = Train_loader,
                                vloader = Val_loader,
                                n_epochs = 25,
                                device = device)
    
    # Save the entire model
    if (where!=''):
        torch.save(model, where)
        
    return model, T_cost_path, V_cost_path