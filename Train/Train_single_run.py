# -*- coding: utf-8 -*-


#Utilities
import os
import time
import torch
from utilities.Dataset import BrainDataset, NeuroData,gather_list_label_file
from utilities.Savemodel import save_model

import matplotlib.pyplot as plt

#Model and training
from Models.DeepBrain import DeepBrain
from Models.resnet18 import S3ConvXFCResnet
from Train.train_func import Training, get_param_dic
from torch.utils.data import random_split


t0 = time.time()
torch.cuda.empty_cache()
dtype=torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#PARAMETERS for single run

DATA="HCP"
model_architecture="resnet18"
model_name_load="fake_train.pth"
#checkpoint_24.pth.tar
model_name_save="test"
hyperparams_file="param.txt"

finetuning=True


dic_FT={"reset": "last_layer",
 "n_output":8,
 "freeze_type":"feature_extractor",
 "drop_out":False
 }

#dic_param_training



#PATH
if DATA=="HCP":
    path_data="../DATA/DATA_HCP/test_code/"



path_data = os.path.abspath(path_data+"DATA_RAW")
path_records = "Records/"
path_records = os.path.abspath(path_records)
path_evaluation = os.path.join(path_records,'metric_training')




dic_param_save = {"path_records":path_records+"\\trained_models",
                "model_name":model_name_save}


## Load hyperparameters

path=os.path.abspath(hyperparams_file)
parame = get_param_dic(path)






if finetuning:
    dic_FT["path_weights"]=os.path.join(path_records+"\\trained_models",model_name_load)


def split(file_list,label_list):
    #TODO
    
    return file_list, label_list, file_list, label_list

#dataset
#split_train_test#TODO
if DATA=="HCP":
    labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]
    file_list, label_list = gather_list_label_file(path_data,labels,extension="nii.gz")
    train_samples, train_labels, val_samples, val_labels = split(file_list,label_list)#TODO

    train_set = BrainDataset(train_samples, train_labels, is_train=True)
    val_set = BrainDataset(val_samples, val_labels, is_train=True)
else:
    labels = ["????"]
    file_list, label_list = gather_list_label_file(path,labels,extension="nii.gz")
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

if model_architecture=="resnet18":
    
    model=S3ConvXFCResnet(27,8)
elif model_architecture=="DeepBrain":
    model=DeepBrain()


if finetuning:    
    model.FT(dic_FT)
else:
    pass

t2 = time.time()






# start the training
train_loader = torch.utils.data.DataLoader(train_set,
                             batch_size=parame.get("batchsize", 3),
                             shuffle=True,
                             num_workers=0)

test_loader = torch.utils.data.DataLoader(val_set,
                             batch_size=parame.get("batchsize", 3),
                             shuffle=True,
                             num_workers=0)




dic_save=dict(dic_param_save)
dic_save["model_name"]+="_best_model"
#Training
Train=Training(model,parame,dic_save,dtype)
model=Train.fit(train_loader,test_loader,save_best_model=True)

history = Train.history      



## Plots


acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(acc) + 1)


plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
path_image=os.path.join(path_evaluation,'f{model_name}/accuracy.png')
plt.savefig(path_image)
plt.close()



plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
path_image=os.path.join(path_evaluation,'f{model_name}/loss.png')
plt.savefig(path_image)




##Save model final_model

dic_save=dict(dic_param_save)
dic_save["model_name"]+="_final_model"
save_model(Train.num_epochs, Train.model, Train.optimizer, Train.criterion, Train.scheduler,dic_save)

#print(history, file=open("outputs/history.txt", "a"))
