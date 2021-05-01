import argparse
import os, time
import torch
import shutil
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import os, glob, logging, distutils, sys
import matplotlib.pyplot as plt
from model_q import resnet20_shiftadd_se
import collections
from collections import OrderedDict

import tqt
import utils
# Training settings
parser = argparse.ArgumentParser(description='PyTorch AdderNet Trainning')
parser.add_argument('--dataset',
                    type=str,
                    default='cifar10',
                    help='training dataset')
parser.add_argument('--data_path',
                    type=str,
                    default=None,
                    help='path to dataset')
parser.add_argument('--batch_size',
                    type=int,
                    default=64,
                    metavar='N',
                    help='batch size for training')
parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    metavar='N',
                    help='batch size for testing')
parser.add_argument('--epoch',
                    type=int,
                    default=160,
                    metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch',
                    type=int,
                    default=0,
                    metavar='N',
                    help='restart point')
parser.add_argument("--quant_lr", default=3e-2, type=float)
parser.add_argument('--lr',
                    type=float,
                    default=1e-4,
                    metavar='LR',
                    help='learning rate')
parser.add_argument("--optimizer",
                    default="SGD",
                    type=str,
                    choices=["Adam", "SGD"])
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
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--grad_clip',
                    type=float,
                    default=5,
                    help='gradient clipping')
parser.add_argument('--freeze', type=int, default=5)
parser.add_argument("--save_folder", default="./sa_quant_results/")
parser.add_argument(
    '--log_interval',
    type=int,
    default=100,
    metavar='N',
    help='how many batches to wait before logging training status')
parser.add_argument("--wb", default=8, type=int)
parser.add_argument("--ab", default=8, type=int)
parser.add_argument("--bb", default=32, type=int)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
save_folder_path = os.path.join(
    args.save_folder, 'dac-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
utils.create_exp_dir(save_folder_path, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_folder_path, 'quant_train_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s", args)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

model = resnet20_shiftadd_se(10)
w = args.wb
b = args.bb
a = args.ab
config = tqt.config.Config(w, b, w, b, a)
tqt.utils.make_net_quant_or_not(model, '', quant=True)
tqt.config.config_network(model,
                          'model',
                          config,
                          show=False,
                          bn_list=[
                              torch.nn.BatchNorm2d, tqt.fold.Conv2dBNReLU,
                              tqt.fold.SA2dBN, tqt.fold.SA2dBNReLU
                          ],
                          conv_linear_list=[
                              torch.nn.Conv2d, tqt.fold.Conv2dBNReLU,
                              tqt.fold.SA2dBN, tqt.fold.SA2dBNReLU,
                              tqt.function.extra.SEConv2d,
                              tqt.function.extra.Adder2d
                          ],
                          acti_list=[
                              torch.nn.ReLU, tqt.fold.Conv2dBNReLU,
                              tqt.fold.SA2dBN, tqt.fold.SA2dBNReLU,
                              tqt.fold.ShareQuant
                          ])
tqt.fold.fold_the_network(model)
model.load_state_dict(torch.load('folded.pth'), strict=True)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
model.cuda()

quant_param = [
    param for name, param in model.named_parameters()
    if name.find('log2') != -1
]
normal_param = [
    param for name, param in model.named_parameters()
    if name.find('log2') == -1
]
if args.optimizer == "SGD":
    optimizer = optim.SGD([{
        'params': quant_param,
        'lr': args.quant_lr
    }, {
        'params': normal_param,
        'lr': args.lr
    }],
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
if args.optimizer == "Adam":
    optimizer = optim.Adam([{
        'params': quant_param,
        'lr': args.quant_lr
    }, {
        'params': normal_param,
        'lr': args.lr
    }],
                           lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       float(args.epoch))
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
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
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        './data.cifar10',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])),
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              **kwargs)
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(
        './data.cifar100',
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
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR100(
        './data.cifar100',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])),
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              **kwargs)

if args.dataset == 'cifar10':
    num_cls = 10
elif args.dataset == 'cifar100':
    num_cls = 100


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def test():
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(
            output, target, size_average=False).item()  # sum up batch loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    test_loss /= len(test_loader.dataset)
    logging.info(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, test_acc, len(test_loader),
            test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)


def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(
                non_blocking=True)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        train_acc += prec1.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            logging.info(
                'Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            # print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))


best_prec1 = 0.
test()
for ep in range(args.start_epoch, args.epoch):
    if ep > args.freeze:
        tqt.utils.make_bn_freeze(model, '', True)
        logging.info('freeze train')
    train(ep)
    prec1 = test()
    scheduler.step()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if is_best:
        torch.save(
            model.state_dict(),
            os.path.join(save_folder_path, 'DACNetquantafterfreeze.pkl'))
