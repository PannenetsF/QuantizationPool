import LeNet_q
import Read
import torch
import torch.nn as nn
from torch.utils.data import dataset, DataLoader
import torch.optim
from torch.autograd import Variable

import argparse

import tqt

parser = argparse.ArgumentParser(description="Simple LeNet")
parser.add_argument("--name", default="LeNet")
parser.add_argument("--type", default="NoQuant")
parser.add_argument("--save_folder", default="./weights/")
args = parser.parse_args()


def train(net, criterion, optimizer, train_loader, epoch):
    net.train()
    train = []
    for idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data /= (255)
        # data -= 1
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # print('------------ {} '.format(len(data)))
        if idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.data.item()))
            train.append(loss.data.item())

    return train


def test(net, criterion, optimizer, train_loader, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        data /= (255.)
        # data -= 1
        output = net(data)
        test_loss = criterion(output, target).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        pred = pred.view_as(target)
        correct += torch.sum(pred.eq(target))

    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Average loss at epoch {}: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
        .format(epoch, test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    # return (correct / len(test_loader.dataset))


net = LeNet_q.LeNet()

train_loader, test_loader = Read.LoadData()

learning_rate = 0.001
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(),
                             learning_rate,
                             betas=(0.9, 0.99))

for i in range(1):
    train(net, criterion, optimizer, train_loader, i)
    test(net, criterion, optimizer, test_loader, i)

tqt.threshold.add_hook(net, '', tqt.threshold.hook_handler)
for idx, (data, target) in enumerate(train_loader):
    data, target = Variable(data), Variable(target)
    break
net(data)
tqt.utils.make_net_quant_or_not(net, '', quant=True)
tqt.threshold.init_network(net, net, '', show=True)

for i in range(5):
    train(net, criterion, optimizer, train_loader, i)
    test(net, criterion, optimizer, test_loader, i)
