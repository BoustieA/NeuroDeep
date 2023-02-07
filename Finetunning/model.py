# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:46:21 2023

@author: 33695
"""


#load model
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

#change classifier+freeze params


#return model


