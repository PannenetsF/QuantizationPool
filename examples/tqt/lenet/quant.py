import LeNet_q as LeNet
import Read
import torch
import torch.nn as nn
from torch.utils.data import dataset, DataLoader
import tqt
import argparse

parser = argparse.ArgumentParser(description="Simple LeNet")
parser.add_argument("--name", default="LeNet")
parser.add_argument("--type", default="NoQuant")
parser.add_argument("--save_folder", default="./weights/")
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--quant_lr", default=0.01)
args = parser.parse_args()


def train(net, criterion, optimizer, train_loader, epoch):
    net.train()
    train = []
    for idx, (data, target) in enumerate(train_loader):
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


train_loader, test_loader = Read.LoadData(50)
net = LeNet.LeNet()
tqt.utils.make_net_quant_or_not(net, 'net', quant=False)
tqt.threshold.add_hook(net, 'net', tqt.threshold.hook_handler, show=True)

img, label = next(iter(test_loader))
net(img)

net.load_state_dict(torch.load('quant9844.pth'))
tqt.utils.make_net_quant_or_not(net, 'net', quant=True)
# tqt.threshold.init_network(net, net, 'net', show=True)
# torch.save(net.state_dict(), 'quant.pth')

learning_rate = 0.001
criterion = nn.CrossEntropyLoss(reduction='sum')
quant_param = [
    param for name, param in net.named_parameters() if name.find('log2') != -1
]
normal_param = [
    param for name, param in net.named_parameters() if name.find('log2') == -1
]
optimizer = torch.optim.Adam([{
    'params': quant_param,
    'lr': args.quant_lr
}, {
    'params': normal_param,
    'lr': args.lr
}],
                             lr=args.lr)

train_loader, test_loader = Read.LoadData(1)
test(net, criterion, optimizer, test_loader, 0)
ans = tqt.threshold.get_hook(net, 'net', show=True)
torch.save(ans, 'every.pth')