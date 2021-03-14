import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import os, shutil


def get_k_hot(data, k, dim):
    topk, indices = torch.topk(data, k, dim=dim)
    out = torch.zeros_like(data)
    out = out.scatter(dim, indices, topk)
    out = (out > 0) * 1.0
    return out


def count_parameters_in_MB(model):
    return np.sum(
        np.prod(v.size()) for name, v in model.named_parameters()
        if "auxiliary" not in name) / 1e6


def count_fm_in_MB(fm):
    return fm.numel() / 1e6


def count_fm_shape_in_MB(fm_shape):
    shape = torch.Size(fm_shape)
    return shape.numel() / 1e6


def load_hyper(model, hyper_path):
    hyper_param = torch.load(hyper_path, map_location='cpu')

    model.beta_normal, model.beta_reduced, model.alpha_chain_normal, model.alpha_chain_reduced, model.alpha_short_normal, model.alpha_short_reduced = hyper_param[
        0], hyper_param[1], hyper_param[2], hyper_param[3], hyper_param[
            4], hyper_param[5]


def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    return res


def data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
