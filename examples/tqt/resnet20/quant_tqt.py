import tqt
from qresnet import resnet18
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
import math
import time
import logging
import os, sys, glob
import utils

parser = argparse.ArgumentParser(description='AdderNet Trainning')
parser.add_argument('--data_path',
                    type=str,
                    default=None,
                    help='path to dataset')
parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    metavar='N',
                    help='batch size for training')
parser.add_argument('--test_batch_size',
                    type=int,
                    default=16,
                    metavar='N',
                    help='batch size for testing')
parser.add_argument('--epochs',
                    type=int,
                    default=50,
                    metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr',
                    type=float,
                    default=1e-6,
                    metavar='LR',
                    help='learning rate')
parser.add_argument('--qlr',
                    type=float,
                    default=1e-2,
                    metavar='qLR',
                    help='qlearning rate')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    metavar='M',
                    help='SGD momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay')
parser.add_argument('--grad_clip',
                    type=float,
                    default=5,
                    help='gradient clipping')
parser.add_argument('--report_freq',
                    type=float,
                    default=50,
                    help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
args = parser.parse_args()
args.save = 'result/exp_' + time.strftime("%Y%m%d-%H%M%S")

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
    './data.cifar10',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])),
                                           batch_size=args.batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
    './data.cifar10',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])),
                                          batch_size=args.test_batch_size,
                                          shuffle=True)

net = resnet18().cuda()
net.load_state_dict(torch.load('resnet18.pth'), strict=False)
loss_function = torch.nn.CrossEntropyLoss()

quant_p = [p for n, p in net.named_parameters() if n.find('log2') != -1]
normal_p = [p for n, p in net.named_parameters() if n.find('log2') == -1]

optimizer = torch.optim.Adam([{
    'params': quant_p,
    'lr': args.qlr
}, {
    'params': normal_p,
    'lr': args.lr
}],
                             args.lr,
                             weight_decay=args.weight_decay)


def train(net, loss_function, optimizer, train_queue, epoch):
    net.train()
    correct_1, correct_5 = 0, 0
    net_loss = torch.tensor(0.)
    bth = args.batch_size
    l = len(train_queue.dataset)
    for idx, (x, t) in enumerate(train_queue):
        x = x.cuda()
        t = t.cuda()
        optimizer.zero_grad()
        out = net(x)
        loss = loss_function(out, t)
        net_loss += loss.data.cpu()
        loss.backward()
        # clip grad or not
        nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()
        res = utils.accuracy(out, t, topk=(1, 5))
        correct_1 += res[0]
        correct_5 += res[1]
        if (idx % args.report_freq == 0):
            logging.info(args.save + ' train epoch %d: %d/%d %f' %
                         (epoch, idx * bth, l, 1. * idx * bth / l))

    logging.info(args.save + ' train epoch %d:' % epoch + ' net loss: %f' %
                 (net_loss))


def valid(net, loss_function, valid_queue, epoch, cali=False):
    net.train()
    correct_1, correct_5 = 0, 0
    net_loss = torch.tensor(0.)
    bth = args.batch_size
    l = len(valid_queue.dataset)
    with torch.no_grad():
        for idx, (x, t) in enumerate(valid_queue):
            x = x.cuda()
            t = t.cuda()
            out = net(x)
            loss = loss_function(out, t)
            net_loss += loss.data.cpu()
            # clip grad or not
            nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            res = utils.accuracy(out, t, topk=(1, 5))
            correct_1 += res[0]
            correct_5 += res[1]
            if cali:
                return
            # if (idx % args.report_freq == 0):
            #     logging.info(args.save + 'epoch %d valid: %d/%d %f' %
            #                  (epoch, idx * bth, l, 1. * idx * bth / l))
    print(correct_1 / len(valid_queue.dataset))


# valid(net, loss_function, test_loader, 0)
# tqt.fold.fold_the_network(net)
# print(net)
tqt.threshold.add_hook(net,
                       'net',
                       tqt.threshold.hook_handler,
                       end_list=[
                           nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d,
                           nn.ReLU6, tqt.fold.ShareQuant
                       ])
valid(net, loss_function, test_loader, 0, cali=True)
tqt.utils.make_net_quant_or_not(net, 'net', quant=True, show=True)
config = tqt.config.Config(8, 16, 8, 8, 8)
tqt.config.config_network(net, '', config)
tqt.threshold.init_network(net, net, 'net', acti_method='max', show=True)
tqt.fold.make_the_shortcut_share(net, show=True)
net.cuda()
tqt.fold.fold_the_network(net)
print(net)
valid(net, loss_function, test_loader, 0, cali=False)
# for i in range(10):
#     train(net, loss_function, optimizer, train_loader, i)
#     valid(net, loss_function, test_loader, 0, cali=False)