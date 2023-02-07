# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:14:17 2022

@author: adywi
"""











import os, fnmatch


import numpy as np
from utilities.Dataset import NeuroData, BrainDataset, gather_list_label_file
from Models.DeepBrain import DeepBrain
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

import matplotlib.pyplot as plt

from Evaluation.Metrics import get_prediction, plot_confusion_matrix
path = os.getcwd()


#PATH
path_data="../DATA/DATA_HCP/test_code/"
read_path = os.path.abspath(path_data+"DATA_RAW")


path_records = "Records/"
path_records = os.path.abspath(path_records)
weights_version='Weights_models/checkpoint_24.pth.tar'
path_weights=os.path.join(path_records,weights_version)




t0 = time.time()

print(torch.cuda.is_available())
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Load Datasets




batch_size=2

path = read_path 
labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]


#dataset
file_list, label_list = gather_list_label_file(path,labels,extension="nii.gz")

test_df=BrainDataset(file_list, label_list, is_train=False)
test_loader = torch.utils.data.DataLoader(test_df,
                            batch_size=batch_size,#batch_size=parameterization.get("batchsize", 3),
                            shuffle=True,
                            num_workers=0)

## load model
model = DeepBrain()
#model.load_state_dict(torch.load(path_weights,
#                                 map_location=torch.device('cpu'))['state_dict'])
model.load_model(path_weights)
#model=DB_LSTM(model_)#lol ça marche, c'est idiot, on a synthétisé les features temporelle via le LSTM pour les duppliquer et les envoyé dans le modèle prétrained de WANG



#prediction
print("predictions")
y_true, y_pred = get_prediction(model,test_loader,dtype,device)

plot_confusion_matrix(y_true, y_pred, labels=None)




