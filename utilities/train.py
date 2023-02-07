# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 20:57:55 2023

@author: 33695
"""




import os, fnmatch
import os
import numpy as np
from utilities.Dataset import NeuroData, BrainDataset, gather_list_label_file
from models.model import DeepBrain
from models.model_lstm import DB_LSTM, RNN_FEATURE_CF
import torch
import torch.optim as optim
import torch.nn as nn
import ax
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate
import time

from utilities.Savemodel import SaveBestModel

import matplotlib.pyplot as plt

from torch.utils.data import random_split

path = os.getcwd()

t0 = time.time()

print(torch.cuda.is_available())
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#dataset
path="DATA_HCP/test_code/"
read_path = os.path.abspath(path+"DATA_RAW")
save_path = os.path.abspath(path+"DATA_transformed_max")





path = read_path 
labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]

file_list, label_list = gather_list_label_file(path,labels,extension="nii.gz")

test_df=BrainDataset(file_list, label_list, is_train=False)


## load model

test_loader = torch.utils.data.DataLoader(test_df,
                            batch_size=2,#batch_size=parameterization.get("batchsize", 3),
                            shuffle=True,
                            num_workers=0)

#model
model=RNN_FEATURE_CF()  



model.to(device)
for batch in iter(test_loader):
    x    = batch[0].to(dtype=dtype, device=device)
    y    = batch[1].type(torch.LongTensor)
    y    = batch[1].to(device=device)
    yhat = torch.argmax(model(x),axis=-1)
    
    break

#training loop
##train function
def net_train(net, train_loader, parameters, dtype, device):
  net.to(dtype=dtype, device=device)

  # Define loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), # or any optimizer you prefer
                        lr=parameters.get("lr", 0.001), # 0.001 is used if no lr is specified
                        momentum=parameters.get("momentum", 0.9)
  )

  scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=int(parameters.get("step_size", 30)),
      gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
  )

  num_epochs = parameters.get("num_epochs", 3) # Play around with epoch number
  # Train Network
  for _ in range(num_epochs):
      for inputs, labels in train_loader:
          # move data to proper dtype and device
          inputs = inputs.to(dtype=dtype, device=device)
          labels = labels.type(torch.LongTensor)
          labels = labels.to(device=device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          scheduler.step()
  return net

def train_evaluate(parameterization):

    # constructing a new training data loader allows us to tune the batch size
    train_loader = torch.utils.data.DataLoader(train_set,
                                batch_size=parameterization.get("batchsize", 3),
                                shuffle=True,
                                num_workers=0)

    test_loader = torch.utils.data.DataLoader(val_set,
                                batch_size=parameterization.get("batchsize", 3),
                                shuffle=True,
                                num_workers=0)

    # Get neural net
    untrained_net = model

    # train
    trained_net = net_train(net=untrained_net, train_loader=train_loader,
                            parameters=parameterization, dtype=dtype, device=device)

    # return the accuracy of the model as it was trained in this run
    return evaluate(
        net=trained_net,
        data_loader=test_loader,
        dtype=dtype,
        device=device,
    )#, trained_net, train_loader, test_loader


#torch.cuda.set_device(0)
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.04], "log_scale": True},
        {"name": "batchsize", "type": "range", "bounds": [10, 64]},
        {"name": "momentum", "type": "range", "bounds": [0.1, 0.99]},
        {"name": "num_epochs", "type": "range", "bounds": [10, 80]},
        {"name": "step_size", "type": "range", "bounds": [20, 40]},
    ],

    total_trials=20,
    evaluation_function=train_evaluate,
    objective_name='accuracy',
)
#save


