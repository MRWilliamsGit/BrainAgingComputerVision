#Maria Williams
#Last Modified: 6/16/22
#Regression prediction script

import torch
import numpy as np

# Function to get a batch predictions
def BatchPredict(model,dataloader,device):
    model = model.to(device) # Send model to GPU if available
    with torch.no_grad():
        model.eval()
        # Get a batch of validation images
        images, labels = iter(dataloader).next()
        images, labels = images.to(device), labels.to(device)
        # Get predictions
        preds = model(images)
        preds = np.squeeze(preds.cpu().numpy())
        preds = np.around(preds, decimals=1)
        labels = labels.cpu().numpy()
    
    return preds, labels

def SinglePredict(model,img,device):
    model = model.to(device) # Send model to GPU if available
    return pred