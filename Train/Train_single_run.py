# -*- coding: utf-8 -*-


#Utilities
import os
import time
import torch
from utilities.Savemodel import save_model
from utilities.PATH import get_torch_datasets

import matplotlib.pyplot as plt

#Model and training
from Models.DeepBrain import DeepBrain
from Models.resnet18 import S3ConvXFCResnet
from Train.train_func_ddp import Training, get_param_dic
from torch.utils.data import random_split


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
#dic_param_training


#PATH
path_records = "Records/"
path_records = os.path.abspath(path_records)

path_records_model=os.path.join(path_records,"trained_models")
path_curves = os.path.join(path_records,'Evaluation\learning curves')
path_history =  os.path.join(path_records,'Evaluation\history')



dic_param_save = {"path_records":path_records_model,
                "model_name":model_name_save}
## Load hyperparameters

path=os.path.abspath(hyperparams_file)
parame = get_param_dic(path)


if finetuning:
    #load parameters for finetuning
    dic_FT=get_param_dic(file_tuning)
    dic_FT["path_weights"]=os.path.join(path_records_model,model_name_load)



#dataset
train_set,val_set = get_torch_datasets(**param_data)


#model selection
if model_architecture=="resnet18":    
    model=S3ConvXFCResnet(27,8)
elif model_architecture=="DeepBrain":
    model=DeepBrain()

#Finetuning preparation of the model
if finetuning:    
    model.FT(dic_FT)


t2 = time.time()






# start the training





dic_save=dict(dic_param_save)
dic_save["model_name"]+="_best_model"
parame["param_save"]=dic_save
#Training
Train=Training(model,parame,dtype)
model=Train.fit(model,train_set,val_set,save_best_model=True)

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
path_image=os.path.join(path_curves,f'{model_name_save}_accuracy.png')
plt.savefig(path_image)
plt.close()



plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
path_image=os.path.join(path_curves,f'{model_name_save}_loss.png')
plt.savefig(path_image)




##Save model final_model

dic_save=dict(dic_param_save)
dic_save["model_name"]+="_final_model"
save_model(Train.num_epochs, Train.model, Train.optimizer, Train.criterion, Train.scheduler,dic_save)

print(history)
