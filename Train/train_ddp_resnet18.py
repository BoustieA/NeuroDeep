# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:19:35 2022

@author: adywi
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler

import os, fnmatch
path = os.getcwd()

import numpy as np
from utilities.Dataset import NeuroData
from utilities.Dataset import BrainDataset

from Models.model import DeepBrain

import torch
import torch.optim as optim
import torch.nn as nn
#from Models.resnet import generate_model

import time

from utilities.Savemodel import SaveBestModel
from Models.resnet18 import S3ConvXFCResnet
import matplotlib.pyplot as plt

from torch.utils.data import random_split

from Train.train_func import *
t0 = time.time()

torch.cuda.empty_cache()

## Load model

model = S3ConvXFCResnet(27,7)


##Destination fichier

labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]

            #'*mental.transformed_max.npy*'
file_list, label_list = gather_list_label_file(read_path,labels,extension="npy")

#TODO split data
train_dataset = BrainDataset(file_list, label_list, is_train=True)
val_dataset = BrainDataset(file_list, label_list, is_train=False)
label_list = np.array(label_list)

print(len(file_list))
print(len(label_list))

def load_data(file_list, label_list):
    train_l = file_list[:20465]
    train_lab = label_list[:20465]
    train2_l = file_list[:150]
    train2_lab = label_list[:150]
    val_l = file_list[20465:23350]
    val_lab = label_list[20465:23350]
    test_l = file_list[23350:29197]
    test_lab = label_list[23350:29197]
    train=NeuroData(train_l, train_lab)
    val=NeuroData(val_l, val_lab)
    test=NeuroData(test_l, test_lab)
    return train, val

train_set, val_set = load_data(file_list,label_list)

print(len(train_set))

torch.backends.cudnn.benchmark=True

import ast


path="param10.txt"
path=os.path.abspath(path)
parame = get_param_dic(path)

#epochs = parame.get("num_epochs", 20) # Play around with epoch number
epochs = 30
# initialize SaveBestModel class

save_best_model = SaveBestModel()


    
if __name__ == "__main__":
    parameters =  parame
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(main, world_size)