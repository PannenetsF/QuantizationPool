import tqt.function as nn
import torch


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.lay1 = nn.Sequential(nn.Conv2dBN(1, 6, 5, padding=2), nn.ReLU6(),
                                  nn.MaxPool2d((2, 2)))
        self.lay2 = nn.Sequential(nn.Conv2dBN(6, 16, 5), nn.ReLU6(),
                                  nn.MaxPool2d((2, 2)))
        self.lay3 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.ReLU6())
        self.lay4 = nn.Sequential(nn.Linear(120, 84), nn.ReLU6())
        self.lay5 = nn.Sequential(nn.Linear(84, 10))

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.lay3(x)
        x = self.lay4(x)
        return self.lay5(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, (2. / n)**0.5)
    elif isinstance(m, nn.BatchNorm2d):
        m.weigth.data.fill_(1)
        m.bias.data.zero_()
