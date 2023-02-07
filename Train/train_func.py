# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:48:23 2023

@author: 33695
"""
import torch
import torch.nn as nn
import torch.optim as optim
from ax.utils.tutorials.cnn_utils import train, evaluate

##train function
def net_train(net, train_loader, parameters, dtype, device):
  net.to(dtype=dtype, device=device)

  # Define loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), # or any optimizer you prefer
                        lr=parameters.get("lr", 0.001), # 0.001 is used if no lr is specified
                        momentum=parameters.get("momentum", 0.9)
  )

  scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=int(parameters.get("step_size", 30)),
      gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
  )

  num_epochs = parameters.get("num_epochs", 3) # Play around with epoch number
  # Train Network
  for _ in range(num_epochs):
      for inputs, labels in train_loader:
          # move data to proper dtype and device
          inputs = inputs.to(dtype=dtype, device=device)
          labels = labels.type(torch.LongTensor)
          labels = labels.to(device=device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          scheduler.step()
  return net

def train_evaluate(parameterization, train_set, val_set, model,
                   dtype,device):

    # constructing a new training data loader allows us to tune the batch size
    train_loader = torch.utils.data.DataLoader(train_set,
                                batch_size=parameterization.get("batchsize", 3),
                                shuffle=True,
                                num_workers=0)

    test_loader = torch.utils.data.DataLoader(val_set,
                                batch_size=parameterization.get("batchsize", 3),
                                shuffle=True,
                                num_workers=0)

    # Get neural net
    untrained_net = model

    # train
    trained_net = net_train(net=untrained_net, train_loader=train_loader,
                            parameters=parameterization, dtype=dtype, device=device)

    # return the accuracy of the model as it was trained in this run
    return evaluate(
        net=trained_net,
        data_loader=test_loader,
        dtype=dtype,
        device=device,
    )#, trained_net, train_loader, test_loader