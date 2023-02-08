# -*- coding: utf-8 -*-

import os, fnmatch
path = os.getcwd()
from torch.autograd import Variable

import numpy as np
from utilities.Dataset import NeuroData, gather_list_label_file
from utilities.Savemodel import save_model
from Models.DeepBrain import DeepBrain
from Models.resnet18 import S3ConvXFCResnet
import torch
import torch.optim as optim
import torch.nn as nn

import time
from utilities.Savemodel import SaveBestModel

import matplotlib.pyplot as plt

from torch.utils.data import random_split

from Train.train_func import *

t0 = time.time()

torch.cuda.empty_cache()


dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#fake train to simulate loading, saving, fine_tunning

finetuning=False
HCP=True


#PATH
path_data="../DATA/DATA_HCP/test_code/"
path_data = os.path.abspath(path_data+"DATA_RAW")

model_name_save="fake_resnet.pth"
model_name_load="fake_resnet.pth"
path_records = "Records/"
path_records = os.path.abspath(path_records)
#weights_version='Weights_models/checkpoint_24.pth.tar'


dic_param_save = {"path_records":path_records+"\\trained_models",
                "model_name":model_name_save}


path_weights=os.path.join(path_records+"\\trained_models",model_name_load)
labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]


#dataset
file_list, label_list = gather_list_label_file(path_data,labels,extension="nii.gz")

## Select samples and smoothed samples so they are in the same set

swar = file_list[0:2]
war = file_list[2:400]

swar_lab = label_list[0:2]
war_lab = label_list[2:400]

train_l = swar[0:1] + war[0:1]
train_lab = swar_lab[0:1] + war_lab[0:1]

print(len(train_l))


val_l = swar[1:2] + war[1:2]
val_lab = swar_lab[1:2] + war_lab[1:2]

## Create sets
if HCP:
    train_set = BrainDataset(train_l, train_lab, is_train=True)
    val_set = BrainDataset(val_l, val_lab, is_train=True)
else:
    train_set = NeuroData(train_l, train_lab)
    val_set = NeuroData(val_l, val_lab)

print(len(train_set))
print(len(val_set))

model=S3ConvXFCResnet(27,8)

import os

if finetuning:
    path_records = "Records/"
    path_records = os.path.abspath(path_records)
    weights_version='trained_models/checkpoint_24.pth.tar'
    path_weights=os.path.join(path_records, weights_version)

    dic={"path_weights": path_weights,
     "reset": "last_layer",
     "n_output":8,
     "freeze_type":"feature_extractor",
     "drop_out":False
     }
    model.FT(dic)
else:
    pass



t2 = time.time()

## Load hyperparameters

path=os.path.abspath("param_fake_train.txt")
parame = get_param_dic(path)


# Define loss and optimizer
criterion, optimizer, scheduler=get_training_tools(model,parame)

epochs = 2 # Play around with epoch number

# initialize SaveBestModel class


# start the training
train_loader = torch.utils.data.DataLoader(train_set,
                             batch_size=parame.get("batchsize", 3),
                             shuffle=True,
                             num_workers=0)

test_loader = torch.utils.data.DataLoader(val_set,
                             batch_size=parame.get("batchsize", 3),
                             shuffle=True,
                             num_workers=0)



def train(model, optimizer, scheduler, loss_fn, train_dl, val_dl, epochs, dtype, device):

    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))
    

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    start_time_sec = time.time()

    model.to(dtype=dtype, device=device)
    for epoch in range(1, epochs+1):
        

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
                                                                                                  
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0
        for batch in train_dl:
            
            
            optimizer.zero_grad()

            x    = batch[0].to(dtype=dtype, device=device)
            y    = batch[1].type(torch.LongTensor)
            y    = batch[1].to(device=device)
            yhat = torch.cat(tuple([y.type(torch.float)[:,None]/i for i in range(1,8+1)]),1)#model(x)


            loss = loss_fn(yhat, y)
            loss = Variable(loss, requires_grad = True)
            #for p in model.parameters():
            #    print(p.requires_grad)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss         += loss.data.item() * x.size(0)


            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            
            num_train_examples += x.shape[0]
            break
        
        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)
        
        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        model.to(device)
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0


        for batch in val_dl:

            x    = batch[0].to(dtype=dtype, device=device)
            y    = batch[1].type(torch.LongTensor)
            y    = batch[1].to(device=device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]
            break
        
        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)
        

        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                (epoch, epochs, train_loss, train_acc, val_loss, val_acc))
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        #save_best_model(val_loss, epoch, model, optimizer, criterion, scheduler)
        #assert False
    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history


history = train(
    model = model,
    optimizer = optimizer,
    scheduler = scheduler,
    loss_fn = criterion,
    train_dl = train_loader,
    val_dl = test_loader, dtype=dtype,
    epochs = epochs,
    device=device)       



## Plots

import matplotlib.pyplot as plt

acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
path_image=os.path.join(path_records,'Evaluation/accuracy.png')
plt.savefig(path_image)
plt.close()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
path_image=os.path.join(path_records,'Evaluation/loss.png')
plt.savefig(path_image)




##Save model
save_model(epochs, model, optimizer, criterion, scheduler,dic_param_save)

#print(history, file=open("outputs/history.txt", "a"))
