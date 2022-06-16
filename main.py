#Maria Williams
#Last Modified: 6/16/22
#All Together (in Cloud9 dev env)

#scripts from repo
from scripts.loaders import RegressionLoaders
from scripts.Rtraining import MakeAndTrain
from scripts.Rpredicting import BatchPredict, SinglePredict

#other imports
import torch
import shutil
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
where = '/home/ec2-user/environment/BrainAgingComputerVision/data'

T, V = RegressionLoaders(where)
print("Loading Complete")

model = MakeAndTrain(T, device)
print("Model Trained")

preds = BatchPredict(model, V, device)
print("Batch Prediction Complete")

pred = SinglePredict(model, img, device)
print("Pred: "+str(pred))

#delete folders afterwards
shutil.rmtree(os.path.join(where, 'train'))
shutil.rmtree(os.path.join(where, 'val'))
shutil.rmtree(os.path.join(where, 'image_data'))