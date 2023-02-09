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
from utilities.PATH import get_torch_datasets

#personnal script

from utilities.Dataset import NeuroData, BrainDataset, gather_list_label_file
from utilities.model_description import get_n_parameters

from Train.train_func_ddp import GS_train_evaluate, get_param_dic

from Models.DeepBrain import DeepBrain
from Models.resnet18 import S3ConvXFCResnet





path = os.getcwd()

t0 = time.time()



print(torch.cuda.is_available())
t0 = time.time()
torch.cuda.empty_cache()
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#PARAMETERS for single run


model_name_save="test"

#model_type
model_architecture="resnet18"#"DeepBrain"

#data choice
param_data={"DATA":"HCP",
            "processing":"raw"}

#hyperparams
hyperparams_file="param.txt"

#FT param
file_tuning = "FT_"+model_architecture+".txt"
finetuning=True
model_name_load="fake_resnet.pth"
#checkpoint_24.pth.tar


#parameters training
parameters_training=[
    {"name": "lr", "type": "range", "bounds": [1e-6, 0.04], "log_scale": True},
    {"name": "batchsize", "type": "range", "bounds": [2, 3]},
    {"name": "momentum", "type": "range", "bounds": [0.1, 0.99]},
    {"name": "num_epochs", "type": "range", "bounds": [10, 80]},#
    {"name": "step_size", "type": "range", "bounds": [20, 40]},
    {"name": "freeze", "type": "choice", "values": ["all","feature_extractore"]},
    
]
dic_param={"DDP":False}
#PATH
path_records = "Records/"
path_records = os.path.abspath(path_records)

path_records_model=os.path.join(path_records,"trained_models")
path_curves = os.path.join(path_records,'Evaluation\learning curves')
path_history =  os.path.join(path_records,'Evaluation\history')



dic_param_save = {"path_records":path_records_model,
                "model_name":model_name_save}


dic_param["param_save"]=dic_param_save

if finetuning:
    #load parameters for finetuning
    dic_FT=get_param_dic(file_tuning)
    dic_FT["path_weights"]=os.path.join(path_records_model,model_name_load)



#dataset




#mapping_dic={0:"all",1:"Feature_extractor"}




#dataset
train_dataset, val_dataset = get_torch_datasets(**param_data)

##Loader unused since in train func(batch size impact training)

## load model

#model
if model_architecture=="resnet18":
    model=S3ConvXFCResnet(27,8)
    get_n_parameters(model)#Find total parameters and trainable parameters
    model=S3ConvXFCResnet#need to be reinstanciate for each train (start from scratch)
elif model_architecture=="DeepBrain":
    model=DeepBrain()
    get_n_parameters(model)#Find total parameters and trainable parameters
    model=DeepBrain#need to be reinstanciate for each train (start from scratch)
    
    







total_trial=2




#torch.cuda.set_device(0)
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






best_parameters, values, experiment, model = optimize(
    parameters=parameters_training,

    total_trials=total_trial,
    evaluation_function = lambda x: GS_train_evaluate(x, train_dataset, val_dataset
                                                 , model, dic_param, dic_FT , True, dtype),
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



render(best_objective_plot)

#find best hyper parameter

data = experiment.fetch_data()
df = data.df
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
best_arm = experiment.arms_by_name[best_arm_name]
print(best_arm)

param=best_arm._parameters
print(param)
print(param, file=open("param.txt", "a"))

#save





