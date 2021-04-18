import torch
import numpy
import LeNet_q as LeNet
import tqt

net = LeNet.LeNet()
tqt.utils.make_net_quant_or_not(net, 'name', quant=True)
weight = torch.load('quant9844.pth')
ans = torch.load('every.pth')
net.load_state_dict(weight)
f = open('quant.txt', 'w')
g = open('out.txt', 'w')
torch.set_printoptions(precision=16, threshold=100000000000000)
numpy.set_printoptions(precision=16, threshold=100000000000000)
for name in weight.keys():
    print(name)
    f.write(str(name) + '\n')
    if name.find('bias') != -1:
        l = 16
    else:
        l = 8
    if name.find('log') != -1:
        f.write(str(weight[name].flatten().numpy()) + '\n\n\n\n')
    else:
        f.write(
            str(
                tqt.function.number.qsigned(weight[name], weight[
                    name + '_log2_t'], l).flatten().numpy()) + '\n\n\n\n')

for n, w in ans:
    print(n)
    g.write(n + '\n')
    g.write(str(w.flatten().detach().numpy()) + '\n\n\n\n')
f.close()
g.close()