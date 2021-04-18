#coding:utf-8

import gzip, struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataset, DataLoader
import torch.optim
from torch.autograd import Variable

data_set_dir = '/home/fyq/Documents/progress/04TQT-LeNet/lenet/data/'

# reference: http://yann.lecun.com/exdb/mnist/


def Read(image_file, lable_file):
    with gzip.open(data_set_dir + lable_file, 'rb') as f:
        magic_num, num_of_item = struct.unpack('>II', f.read(8))
        lable = np.fromstring(f.read(), dtype=np.uint8)
    with gzip.open(data_set_dir + image_file, 'rb') as f:
        magic_num, num_of_item, num_of_rows, num_of_cols = struct.unpack(
            '>IIII', f.read(16))
        image = np.fromstring(f.read(), dtype=np.uint8)
        image = image.reshape(len(lable), num_of_rows, num_of_cols)
    return lable, image


def ReadTrain():
    return Read('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')


def ReadTest():
    return Read('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')


def LoadData(batch_size=16):
    train_lable, train_image = ReadTrain()
    test_lable, test_image = ReadTest()

    # reshep image set to (num, 1, 28, 28)
    train_image, train_lable = torch.from_numpy(
        train_image.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(
            train_lable.astype(int))
    test_image, test_lable = torch.from_numpy(test_image.reshape(
        -1, 1, 28, 28)).float(), torch.from_numpy(test_lable.astype(int))

    train_set = dataset.TensorDataset(train_image, train_lable)
    test_set = dataset.TensorDataset(test_image, test_lable)

    kwargs = {'num_workers': 2, 'pin_memory': True}

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              **kwargs)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=True,
                             **kwargs)

    return train_loader, test_loader
