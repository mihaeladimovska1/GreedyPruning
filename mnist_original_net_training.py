'''We train a small network for 10 epochs on the MNIST data set. We extract the parameters from the 10th epoch. '''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import numpy as np
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Training settings
batch_size = 128
total_num_epochs = 10

#list that will hold our parameters
layers_parameters = []

# EMNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=9)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=9)
        self.conv3 = nn.Conv2d(10, 8, kernel_size=8)
        self.conv4 = nn.Conv2d(8, 5, kernel_size=1)
        self.fc1 = nn.Linear(5*5*5, 10)
        self.lsm = nn.LogSoftmax()

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        #print(x.shape)
        x = x.view(-1, 5*5*5)
        x = self.fc1(x)

        return self.lsm(x)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        count+=1
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            output = model(data)

        # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).data.item()
        # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1,total_num_epochs+1):
    train(epoch)
    test()
    if epoch==10:

        l1 = list(model.conv1.parameters())
        layers_parameters.append(l1)
        l2 = list(model.conv2.parameters())
        layers_parameters.append(l2)
        l3 = list(model.conv3.parameters())
        layers_parameters.append(l3)
        l4 = list(model.conv4.parameters())
        layers_parameters.append(l4)
        l5 = list(model.fc1.parameters())
        layers_parameters.append(l5)

        fw = open('parameters_original_net/parameters_epoch_'+str(epoch), 'wb')
        pickle.dump(layers_parameters, fw)
        fw.close()

