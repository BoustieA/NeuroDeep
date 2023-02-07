# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:58:19 2023

@author: 33695
"""



import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def get_prediction(model,dataset_loader,dtype,device):
    y_pred=[]
    y_true=[]
    model.eval()
    model.to(device)
    for batch in iter(dataset_loader):
        x    = batch[0].to(dtype=dtype, device=device)
        y    = batch[1].type(torch.LongTensor)
        y    = batch[1].to(device=device)
        yhat = torch.argmax(model(x),axis=-1)
        
        y_true+=y.tolist()
        y_pred+=yhat.tolist()
    return y_true, y_pred



def plot_confusion_matrix(y_true, y_pred, labels=None):
    CM=confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(CM,annot=True)
    plt.title("Matrice de confusion")
    plt.plot()
