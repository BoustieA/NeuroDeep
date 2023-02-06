# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 21:03:34 2023

@author: 33695
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 03:13:48 2023

@author: 33695
"""
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
LSTM=nn.LSTM(5,1,1)

X=torch.rand(2,5)
print(LSTM(X))
print()
LSTM=nn.LSTM(5,1,1)

X=torch.rand(4,2,5)
print(LSTM(X))

print("#"*10)
X=torch.rand(2,4,4,4,5)
print(X.shape)
X=X.view(2,4**3,5)
print(X.shape)
LSTM=nn.LSTM(5,1,1)
Y=LSTM(X)[0]
print(Y.shape)


X=torch.rand(16,16,5)
#plt.imshow(X.view(16,16))
#plt.figure()
#X=torch.cat((X,1/(1+X),3*X,4*X,5*X),-1)


for i in X.transpose(0,-1).numpy():
    print(i.shape)
    plt.imshow(i)
    plt.figure()
assert False
print(X.shape)
X=X.view(16**2,5).transpose(0,-1)
print(X.shape)


LSTM=nn.LSTM(256,256,1)
Y=LSTM(X)[0]
print(Y.shape)
plt.figure()
plt.plot()
for i in range(5):
    plt.figure()
    plt.imshow(Y.view(5,16,16).detach().numpy()[i,:,:])



"""
LSTM input:
    3D tensor (sequence,item_batch,features)
    
"""