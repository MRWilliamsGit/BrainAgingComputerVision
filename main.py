#Maria Williams
#Last Modified: 6/16/22
#All Together (in Cloud9 dev env)

#scripts from repo
from scripts.loaders import RegressionLoaders, SingleLoader
from scripts.Rtraining import MakeAndTrain
from scripts.Rpredicting import BatchPredict, SinglePredict

#other imports
import torch
import shutil
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
where = '/home/ec2-user/environment/BrainAgingComputerVision/data'

#T, V = RegressionLoaders(where)
print("Loading Complete")

#model = MakeAndTrain(T, device)
print("Model Trained")

#preds = BatchPredict(model, V, device)
print("Batch Prediction Complete")

model2 = torch.load('/home/ec2-user/environment/BrainAgingComputerVision/models/model.pt')
img = np.load('/home/ec2-user/environment/BrainAgingComputerVision/data/example0.npy')
image = SingleLoader(img)
pred = SinglePredict(model2, image, device)
print("Pred: "+str(pred))

#delete folders afterwards
#shutil.rmtree(os.path.join(where, 'train'))
#shutil.rmtree(os.path.join(where, 'val'))
#shutil.rmtree(os.path.join(where, 'image_data'))