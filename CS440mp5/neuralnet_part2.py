# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP5. You should only modify code
within this file and neuralnet_part1.py,neuralnet_leaderboard -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """

        super().__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.conv1 = nn.Conv2d(3, 26, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(26, 48, 5)
        self.fc1 = nn.Linear(48 * 4 * 4, 72)
        self.fc2 = nn.Linear(72, 120)
        self.fc3 = nn.Linear(120, out_size)
        #print(out_size)
        self.optimizer = optim.SGD(self.parameters(), lr=self.lrate)

        
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        x = x.view(x.shape[0], 3, 31, 31)
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.forward(x)


        
        loss = self.loss_fn(outputs, y)
        # print(loss.item())
        loss.backward()
        self.optimizer.step()

        # print statistics
        return float(loss.detach().cpu().numpy())

def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # construct a NeuralNet object
    nn1 = NeuralNet(0.15, nn.CrossEntropyLoss(), 2883, 4)

    dataset_mean = train_set.mean()
    # torch.reshape(torch.sum(train_set, dim=0) / train_set.shape[0], (1, train_set.shape[1]))
    dataset_std = train_set.std()
    # torch.reshape(torch.var(train_set, dim=0)/ train_set.shape[0], (1, train_set.shape[1]))

    train_set = (train_set - dataset_mean) / dataset_std
    dev_set = (dev_set - dataset_mean) / dataset_std



    #converts tensors of features (X) and labels (Y) into a simple torch dataset class that can be loaded into their dataloaders
    dataset = get_dataset_from_arrays(train_set, train_labels)
    

    trainloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        
        for data in trainloader:
            inputs = data['features']
            labels = data['labels']
            
            running_loss += nn1.step(inputs, labels)

        losses.append(running_loss / batch_size)
        #print(running_loss / batch_size)

    
    outputs = nn1.forward(dev_set).detach().cpu().numpy()

    yhats = []
    for probs in outputs:
        #print(probs)
        yhats.append(np.argmax(probs))

    y_ = np.asarray(yhats)
    return losses,y_.astype(int),nn1

