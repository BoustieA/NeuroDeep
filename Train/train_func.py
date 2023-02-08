# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 12:48:23 2023

@author: 33695
"""
import torch
import torch.nn as nn
import torch.optim as optim
from utilities.Savemodel import SaveBestModel, save_model
import ast
import time

def get_param_dic(path):
    parame = open(path, "r")
    parame = parame.read()
    return ast.literal_eval(parame)

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
class Training:
    """
    class for trainning
    
    Define training and evaluation steps,
    
    
    """
    def __init__(self,model,parameters,dic_param_save,dtype):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dtype=dtype
        
        self.num_epochs = parameters.get("num_epochs",1)
        self.distributed = parameters.get("DDP",False)
        self.save_best_model=SaveBestModel(dic_param_save)
        
        self.model=model
        self.model.to(dtype=self.dtype, device=self.device)
        criterion, optimizer, scheduler = get_training_tools(model,parameters)
        self.criterion=criterion
        self.optimizer=optimizer
        self.scheduler=scheduler
        
    
        self.init_history()
    
    def init_history(self):
        history = {} # Collects per-epoch loss and acc like Keras' fit().
        history['loss'] = []
        history['val_loss'] = []
        history['acc'] = []
        history['val_acc'] = []
        self.history=history
        
    def train_step(self, train_loader, rank=None):
        self.model.train()
        train_loss       = 0.0
        num_train_correct  = 0
        num_train_examples = 0
        
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            if self.distributed:
                inputs.cuda(non_blocking=True)
                inputs = inputs.to(dtype=self.dtype)
                inputs = inputs.to(rank)
                inputs.cuda(non_blocking=True)
                labels = labels.to(dtype=self.dtype)
                labels = labels.to(rank)
            else:
                inputs = inputs.to(dtype=torch.float, device=self.device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device=self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            if rank == 0:
                save_model(epoch, self.model,self.optimizer, self.criterion, self.scheduler)
            #metrics
            train_loss         += loss.data.item() * inputs.size(0)
            num_train_correct  += (torch.max(outputs, 1)[1] == labels).sum().item()
            num_train_examples += inputs.shape[0]
            train_acc   = num_train_correct / num_train_examples
            train_loss  = train_loss / len(train_loader.dataset)
            
            self.history['loss'].append(train_loss)
            self.history['acc'].append(train_acc)
    
    
    def validation_step(self,val_loader,rank):
        self.model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for i, (images, labels) in enumerate(val_loader):

            if self.distributed:
                images = images.cuda(non_blocking=True)
                images = images.to(self.dtype)
                images = images.to(rank)
                labels = labels.cuda(non_blocking=True)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(rank)
            else:
                images = images.to(dtype=self.dtype, device=self.device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device=self.device)
                
          # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            val_loss         += loss.data.item() * images.size(0)
            num_val_correct  += (torch.max(outputs, 1)[1] == labels).sum().item()
            num_val_examples += images.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_loader.dataset)
        
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        
    
    def fit(self,train_loader,val_loader=None,rank=None,verbose=True,save_best_model=False):
        """
        

        Parameters
        ----------
        train_loader : torch loader
            trainning data
        val_loader : torch loader, optional
            validation data or test data regarding the needs
        rank : int, optional
            arguments for distributed computing on GPU
        verbose : Bool, optional
            same as keras
        save_best_model : Bool, optional
            save the model at each epoch if validation result is better than the previous one

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
              (type(self.model).__name__, type(self.optimizer).__name__,
               self.optimizer.param_groups[0]['lr'], self.num_epochs, self.device))
        start_time_sec = time.time()    
        for epoch in range(self.num_epochs):
            self.train_step(train_loader,rank)
            if val_loader is not None:
                self.validation_step(val_loader,rank)
                self.save_best_model(self.history["val_loss"][-1], epoch
                                     , self.model, self.optimizer, self.criterion, self.scheduler)
            else:
                assert not save_best_model, "saving while training need a validation set to compare models"
            if verbose:
                print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                    (epoch, self.num_epochs, self.history["loss"][-1], self.history["acc"][-1]
                     , self.history["val_loss"][-1], self.history["val_acc"][-1]))
            
        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.num_epochs
        print()
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
        return self.model
    
    def evaluate(self,test_loader,rank):
        """
        reset history of training then return the evaluation_dic
        """
        self.validation_step(test_loader,rank)
        return self.history["val_acc"][-1]

    


def GS_train_evaluate(parameterization, train_set, val_set, model,
                   finetuning=False, dtype=torch.float):
    """
    /!\ model need not to be instancied so it can be reseted
    """
    #loader input loader and change batch size#TODO
    _model = model()
    if finetuning:
        _model.FT(parameterization)
        
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
    Train=Training(_model,parameterization,dtype)
    _model=Train.fit(train_loader) 

    # train

    # return the accuracy of the model as it was trained in this run
    return Train.evaluate(test_loader)#, trained_net, train_loader, test_loader






















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
