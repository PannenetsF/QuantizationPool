import torch
import wageubn
from resnet20 import resnet20


def load_pretrained():
    dic = torch.load('ResNet20-AdderNet.pth')
    allkey = []
    allval = []
    sigkey = []
    for key in dic.keys():
        opkey = key[7:]
        allkey.append((opkey.replace('adder', 'weight'), key))
        allval.append(dic[key])
    newdic = dic.fromkeys(sigkey)
    for oped, ored in allkey:
        newdic[oped] = dic[ored]

    net = resnet20()
    net.load_state_dict(newdic, strict=True)
    config = wageubn.config.Config(conv_and_linear_weight=8,
                                   conv_and_linear_bias=8,
                                   bn_weight=8,
                                   bn_bias=8,
                                   bn_mean=16,
                                   bn_var=8,
                                   bn_out=8,
                                   acti=8)
    wageubn.config.config_network(net, 'net', config, show=True)
    return net


if __name__ == '__main__':
    load_pretrained()