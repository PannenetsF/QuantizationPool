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

import matplotlib.pyplot as plt
from model_q import resnet20_shiftadd_se
import collections
from collections import OrderedDict
import tqt

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
                    default=128,
                    metavar='N',
                    help='batch size for training')
parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    metavar='N',
                    help='batch size for testing')
parser.add_argument('--epochs',
                    type=int,
                    default=160,
                    metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch',
                    type=int,
                    default=0,
                    metavar='N',
                    help='restart point')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='learning rate schedule')
parser.add_argument('--lr',
                    type=float,
                    default=0.1,
                    metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-sign',
                    default=None,
                    type=float,
                    help='separate initial learning rate for sign params')
parser.add_argument('--lr_decay',
                    default='stepwise',
                    type=str,
                    choices=['stepwise', 'poly'])
parser.add_argument('--optimizer',
                    type=str,
                    default='sgd',
                    help='used optimizer')
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
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed')
parser.add_argument('--save',
                    default='./logs',
                    type=str,
                    metavar='PATH',
                    help='path to save prune model')
parser.add_argument('--save_suffix',
                    default='retrain_',
                    type=str,
                    help='identify which retrain')
parser.add_argument('--arch',
                    default='resnet20',
                    type=str,
                    help='architecture to use')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument(
    '--log_interval',
    type=int,
    default=100,
    metavar='N',
    help='how many batches to wait before logging training status')
# multi-gpus
parser.add_argument('--gpu_ids',
                    type=str,
                    default='0',
                    help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# shift hyper-parameters
parser.add_argument('--shift_depth',
                    type=int,
                    default=0,
                    help='how many layers to convert to shift')
parser.add_argument('--shift_type',
                    type=str,
                    choices=['Q', 'PS'],
                    help='shift type for representing weights')
parser.add_argument('--rounding',
                    default='deterministic',
                    choices=['deterministic', 'stochastic'])
parser.add_argument('--weight_bits',
                    type=int,
                    default=5,
                    help='number of bits to represent the shift weights')
parser.add_argument('--use-kernel',
                    type=lambda x: bool(distutils.util.strtobool(x)),
                    default=False,
                    help='whether using custom shift kernel')
# add hyper-parameters
parser.add_argument('--add_quant',
                    type=bool,
                    default=False,
                    help='whether to quantize adder layer')
parser.add_argument('--add_bits',
                    type=int,
                    default=8,
                    help='number of bits to represent the adder filters')
# visualization
parser.add_argument('--visualize',
                    action="store_true",
                    default=False,
                    help='if use visualization')
# reinit
parser.add_argument('--reinit',
                    type=str,
                    default=None,
                    help='whether reinit or finetune')
parser.add_argument("--wb", default=32, type=int)
parser.add_argument("--ab", default=32, type=int)
parser.add_argument("--bb", default=32, type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

cudnn.benchmark = True

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
        torch.save(data, 'data.pth')
        raise Exception('')
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if args.visualize:
            _, output = model(data)
        else:
            output = model(data)
        test_loss += F.cross_entropy(
            output, target, size_average=False).item()  # sum up batch loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()
    test_loss /= len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, test_acc, len(test_loader),
            test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)


if __name__ == '__main__':
    model = resnet20_shiftadd_se(10)
    tqt.utils.make_net_quant_or_not(model, 'net', quant=False)
    data = torch.load('model.pth')
    data_new = {}
    for k in data.keys():
        data_new[k.replace('adder', 'weight')] = data[k]
    data_new = OrderedDict(data_new)
    model.load_state_dict(data_new, strict=False)
    test_batch = torch.load('data.pth')
    print('init', model(test_batch))
    w = args.wb
    b = args.bb
    a = args.ab
    config = tqt.config.Config(w, b, w, b, a)
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
    tqt.threshold.add_hook(model,
                           '',
                           tqt.threshold.hook_handler,
                           end_list=[
                               tqt.function.extra.Adder2d,
                               tqt.function.extra.SEConv2d, nn.ReLU,
                               nn.BatchNorm2d, tqt.fold.ShareQuant
                           ],
                           show=False)
    model(test_batch)
    tqt.utils.make_net_quant_or_not(model, 'net', quant=True, exclude=[])
    tqt.threshold.init_network(model,
                               model,
                               '',
                               weight_method='max',
                               bias_method='max',
                               acti_method='max',
                               show=False)
    model.cpu()
    tqt.fold.make_the_shortcut_share(model)
    tqt.fold.fold_the_network(model)
    tqt.utils.make_net_quant_or_not(model, '', quant=True)
    print('fold_q', model(test_batch))
    torch.save(model.state_dict(), 'folded.pth')
