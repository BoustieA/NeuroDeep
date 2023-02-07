# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 20:57:55 2023

@author: 33695
"""




import os, fnmatch
import os
import numpy as np

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


#personnal script

from utilities.Dataset import NeuroData, BrainDataset, gather_list_label_file
from utilities.model_description import get_n_parameters

from Train.train_func import net_train, train_evaluate

from models.model import DeepBrain





path = os.getcwd()

t0 = time.time()



print(torch.cuda.is_available())
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#path data
path = "../DATA/DATA_HCP/test_code/"
read_path = os.path.abspath(path+"DATA_RAW")

#parameters training
parameters_training=[
    {"name": "lr", "type": "range", "bounds": [1e-6, 0.04], "log_scale": True},
    {"name": "batchsize", "type": "range", "bounds": [2, 3]},
    {"name": "momentum", "type": "range", "bounds": [0.1, 0.99]},
    {"name": "num_epochs", "type": "range", "bounds": [10, 80]},
    {"name": "step_size", "type": "range", "bounds": [20, 40]},
]






labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]






#dataset
file_list, label_list = gather_list_label_file(read_path,labels,extension="nii.gz")


train_dataset = BrainDataset(file_list, label_list, is_train=True)
val_dataset = BrainDataset(file_list, label_list, is_train=False)
##Loader unused since in train func(batch size impact training)

## load model

#model
model=DeepBrain()  
#model=DeepBrain()
#Find total parameters and trainable parameters
get_n_parameters(model)

model.to(device)


#training loop












#torch.cuda.set_device(0)
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






best_parameters, values, experiment, model = optimize(
    parameters=parameters_training,

    total_trials=20,
    evaluation_function=lambda x: train_evaluate(x, train_dataset, val_dataset
                                                 , model, dtype, device),
    objective_name='accuracy',
)


print(best_parameters)
means, covariances = values
print(means)
print(covariances)





#Plot accuracy

best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])

best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Classification Accuracy, %",
)

assert False

render(best_objective_plot)

#find best hyper parameter

data = experiment.fetch_data()
df = data.df
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
best_arm = experiment.arms_by_name[best_arm_name]
print(best_arm)

parame=best_arm._parameters
print(param)
print(param, file=open("param.txt", "a"))

#save























"""
lambda x :train_evaluate(x
                        , train_set, val_set, model
                        , dtype,device),
"""



assert False





















file_list = []

for path, folders, files in os.walk(path):
    for file in files:
        if fnmatch.fnmatch(file, '*train.npy'):
            file_list.append(os.path.join(path, file))

print(len(file_list))
file_list.sort()

label_list = []

for file in file_list:
    if fnmatch.fnmatch(file, '*gene_train.npy*' ):
            label = 0
            label_list.append(label)
    elif fnmatch.fnmatch(file, '*rap_train.npy*' ):
                label = 1
                label_list.append(label)


## Select samples and smoothed samples so they are in the same set

swar = file_list[0:200]
war = file_list[200:400]

swar_lab = label_list[0:200]
war_lab = label_list[200:400]

train_l = swar[0:140] + war[0:140]
train_lab = swar_lab[0:140] + war_lab[0:140]

print(len(train_l))


val_l = swar[140:160] + war[140:160]
val_lab = swar_lab[140:160] + war_lab[140:160]

## Create sets

train_set = NeuroData(train_l, train_lab)
val_set = NeuroData(val_l, val_lab)

print(len(train_set))
print(len(val_set))

##Define model
model = DeepBrain()
model.load_state_dict(torch.load('./models/checkpoint_24.pth.tar')['state_dict'])


##Freeze param
for param in model.parameters():
    param.requires_grad = False

##Recreate FC layers
model.classifier = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 2),
    nn.LogSoftmax(dim=1))

#Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

##train function




