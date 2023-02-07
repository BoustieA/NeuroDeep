# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:48:23 2023

@author: 33695
"""
import torch
import torch.nn as nn
import torch.optim as optim
from ax.utils.tutorials.cnn_utils import train, evaluate

import ast


def get_param_dic(path):
    parame = open(path, "r")
    parame = parame.read()
    parame = ast.literal_eval(parame)


def get_training_tools(model,parame):
    #return the needed object for training
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), # or any optimizer you prefer
                            lr=parame.get("lr", 0.001), # 0.001 is used if no lr is specified
                            momentum=parame.get("momentum", 0.9)
      )

    scheduler = optim.lr_scheduler.StepLR(
          optimizer,
          step_size=int(parame.get("step_size", 30)),
          gamma=parame.get("gamma", 1.0),  # default is no learning rate decay
      )

    return criterion, optimizer, scheduler

##train function
def net_train(net, train_loader, parameters, dtype, device):
  net.to(dtype=dtype, device=device)

  # Define loss and optimizer
  criterion, optimizer, scheduler = get_training_tools(net,parameters)
  
  
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






















#DDP


def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    dataset = dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False, sampler=sampler)

    return dataloader


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    
##Save model
def save_model(epochs, model, optimizer, criterion, scheduler):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'model_state_dict': model.state_dict(),
                'loss': criterion,
                }, 'outputs/checkpoint.pth')

def main(model,parameters, rank, world_size):

    dtype=torch.float
    criterion, optimizer, scheduler = get_training_tools(model,parameters)
    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs))

    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    train_loader = prepare(rank, world_size, train_set, batch_size=38)
    val_loader = prepare(rank, world_size, val_set, batch_size=38)

    # instantiate the model(it's your own model) and move it to the right device
    modelddp = model.to(rank)
    modelddp = modelddp.to(dtype)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    modelddp = DDP(modelddp, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    #################### The above is defined previously

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []


    start_time_sec = time.time()

    total_step = len(train_loader)

    for epoch in range(epochs):

        modelddp.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for i, (images, labels) in enumerate(train_loader):
           images = images.cuda(non_blocking=True)
           images = images.to(dtype)
           images = images.to(rank)
           labels = labels.cuda(non_blocking=True)
           labels = labels.type(torch.LongTensor)
           labels = labels.to(rank)

          # Forward pass
           outputs = modelddp(images)
           loss = criterion(outputs, labels)

           # Backward and optimize
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           scheduler.step()

           if rank == 0:
               save_model(epoch, modelddp,optimizer, criterion, scheduler)

           train_loss         += loss.data.item() * images.size(0)
           num_train_correct  += (torch.max(outputs, 1)[1] == labels).sum().item()
           num_train_examples += images.shape[0]


        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_loader.dataset)



       # --- EVALUATE ON VALIDATION SET -------------------------------------
        modelddp.eval()

        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for i, (images, labels) in enumerate(val_loader):


           images = images.cuda(non_blocking=True)
           images = images.to(dtype)
           images = images.to(rank)
           labels = labels.cuda(non_blocking=True)
           labels = labels.type(torch.LongTensor)
           labels = labels.to(rank)

          # Forward pass
           outputs = modelddp(images)
           loss = criterion(outputs, labels)

           val_loss         += loss.data.item() * images.size(0)
           num_val_correct  += (torch.max(outputs, 1)[1] == labels).sum().item()
           num_val_examples += images.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_loader.dataset)


        
        if rank == 0:
            print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            save_best_model(val_loss, epoch, modelddp, optimizer, criterion, scheduler)
            print(epoch, file=open("outputs/epochs.txt", "a"))
            print(history, file=open("outputs/history.txt", "a"))


            acc = history['acc']
            print(acc, file=open("outputs/acc.txt", "a"))

            val_acc = history['val_acc']
            print(val_acc, file=open("outputs/val_acc.txt", "a"))

            loss = history['loss']
            print(loss, file=open("outputs/loss.txt", "a"))

            val_loss = history['val_loss']
            print(val_loss, file=open("outputs/val_loss.txt", "a"))


        # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    cleanup()
