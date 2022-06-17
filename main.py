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
where = '/home/ec2-user/environment/BrainAgingComputerVision'

T, V = RegressionLoaders(os.path.join(where, 'sampl_data'))
print("Loading Complete")

model, T_cost_path, V_cost_path = MakeAndTrain(T, V, device)
print("Model Trained")

preds = BatchPredict(model, V, device)
print("Batch Prediction Complete")

#model2 = torch.load(os.path.join(where, 'models/model.pt'))
#img = np.load(os.path.join(where, 'data/example0.npy'))

#image = SingleLoader(img)
#pred = SinglePredict(model2, image, device)
#print("Pred: "+str(pred))

#delete folders afterwards
shutil.rmtree(os.path.join(where, 'sampl_data/train'))
shutil.rmtree(os.path.join(where, 'sampl_data/val'))
shutil.rmtree(os.path.join(where, 'sample_data/image_data'))