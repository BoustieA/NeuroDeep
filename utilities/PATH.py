# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:25:07 2023

@author: 33695
"""
import numpy as np

from utilities.Dataset import NeuroData, NeuroDataset, BrainDataset, BrainDataset2
import os, fnmatch


def gather_list_label_file(read_path,labels,extension="nii.gz"):
    """
    

    Parameters
    ----------
    read_path : string
        path of the folder containing the files
    labels : list of string
        DESCRIPTION.
    extension : string, optional
        extension of the files. 
        The default is "nii.gz".


    Returns
    -------
    file_list : list of string
        list of absolute path toward the sample of data
        
    label_list: list of string
        list of labels extracted from the file_name.

    """
    file_list=[]
    label_list=[]
    dic_label={label : i for i,label in enumerate(labels) }
    for path, folders, files in os.walk(read_path):
        for file in files:
            if fnmatch.fnmatch(file, '*'+extension):
                file_list.append(os.path.join(read_path,file))
                for label in labels:
                    if fnmatch.fnmatch(file, '*'+label+'*'+extension ):
                            label_list.append(dic_label[label])
    return file_list, label_list



def get_files_label(DATA="HCP",processing="max"):
    if DATA == "HCP":
        path_data = "../DATA/DATA_HCP/test_code/"
        labels = ["EMOTION","GAMBLING","LANGUAGE","MOTOR","RELATIONAL","SOCIAL","WM"]
        if processing=="max":
            path_data+="DATA_PREPROCESSED/"
            extension="npy"
            pass
        elif processing=="raw":
            path_data+="DATA_RAW/"
            extension="nii.gz"
    elif DATA == "INLANG":
        path_data="../DATA/DATA_INLANG/"
        labels = ["gene","rap"]
        if processing=="max":
            path_data+="DATA_PREPROCESSED/"
            extension="nii.gz"
            pass
        elif processing=="raw":
            path_data+="DATA_RAW/"
            extension="npy"
    return gather_list_label_file(path_data,labels,extension)
    


def split_train_test_HCP(file_list,label_list):
    return file_list,label_list, file_list, label_list

def split_train_test_INLANG(file_list,label_list):
    war_list=[]
    swar_list=[]
    war_label=[]
    swar_label=[]
    for i,file in enumerate(file_list):
        if fnmatch.fnmatch(file,'*'+"swar"+ '*'):
            war_list+=[file]
            war_label+=[label_list[i]]
        elif fnmatch.fnmatch(file,'*'+"war"+ '*'):
            swar_list+=[file]
            swar_label+=[label_list[i]]
    
    


    swar = file_list[0:200]
    war = file_list[200:400]
    
    swar_lab = label_list[0:200]
    war_lab = label_list[200:400]
    
    train_l = swar[0:140] + war[0:140]
    train_lab = swar_lab[0:140] + war_lab[0:140]
    
    val_l = swar[140:160] + war[140:160]
    val_lab = swar_lab[140:160] + war_lab[140:160]
    
    return train_l,train_lab,val_l,val_lab 


def get_torch_datasets(DATA="HCP",processing="max"):
    file_list, label_list  = get_files_label(DATA=DATA,processing=processing)
    if DATA=="HCP":
        train_l,train_lab,val_l,val_lab  = split_train_test_HCP(file_list, label_list)
        if processing=="max":
            train_set = BrainDataset2(train_l, train_lab, is_train=True)
            val_set = BrainDataset2(val_l, val_lab, is_train=True)
        elif processing=="raw":
            train_set = BrainDataset(train_l, train_lab, is_train=True)
            val_set = BrainDataset(val_l, val_lab, is_train=True)
    elif DATA=="INLANG":
        train_l,train_lab,val_l,val_lab  = split_train_test_INLANG(file_list, label_list)
        if processing=="max":
            train_set = NeuroData(train_l, train_lab)
            val_set = NeuroData(val_l, val_lab)
        else:
            train_set = NeuroDataset(train_l, train_lab)
            val_set = NeuroDataset(val_l, val_lab)
    
    return train_set, val_set
    
T,V=get_torch_datasets(DATA="HCP",processing="raw")