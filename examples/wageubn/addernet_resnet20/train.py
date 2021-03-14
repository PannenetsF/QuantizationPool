import torch
import torch.nn as nn
import load
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
                    default=0.1,
                    metavar='LR',
                    help='learning rate')
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
writer = SummaryWriter(args.save)

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

writer = SummaryWriter(args.save)
torch.cuda.set_device(args.gpu)
net = load.load_pretrained().cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

schedule_cosine_lr_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, args.epochs, eta_min=0, last_epoch=-1)


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

    writer.add_scalars('Network Train', {
        'Loss': net_loss,
        'Top1': correct_1 / l,
        'Top5': correct_5 / l
    }, epoch)
    logging.info(args.save + ' train epoch %d:' % epoch + ' net loss: %f' %
                 (net_loss))


def valid(net, loss_function, optimizer, valid_queue, epoch):
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
            if (idx % args.report_freq == 0):
                logging.info(args.save + 'epoch %d valid: %d/%d %f' %
                             (epoch, idx * bth, l, 1. * idx * bth / l))
    print(correct_1 / len(valid_queue.dataset))
    # writer.add_scalars(
    #     'Network', {
    #         'VLoss': net_loss,
    #         'VTop1': correct_1 / len(valid_queue.dataset),
    #         'VTop5': correct_5 / len(valid_queue.dataset)
    #     }, epoch)


for epoch in range(args.epochs):
    schedule_cosine_lr_decay.step(epoch)
    # train(net, loss_function, optimizer, train_loader, epoch)
    valid(net, loss_function, optimizer, test_loader, epoch)
# torch.save(net.state_dict(), args.save + 'model.pth')