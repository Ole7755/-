import torch
import torch.nn as nn
import torch.nn.functional as F

"""
构建LeNet并采用步进(Step into)的调试方法从创建网络模型开始（net = LeNet(classes=2)）进入到每一个被调用函数，观察net的_modules字段何时被构建并且赋值，记录其中所有进入的类与函数
"""


class LeNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)
        self.initialize_weights()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        # view() 是 PyTorch 中改变张量（Tensor）形状的方法。它不会改变数据本身.
        # 保留批次维度不变，把剩下的所有特征全部拉平。view后的张量围度(out.size(0),c*h*w)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        #  self.modules() 是 PyTorch 的一个内置方法，它会递归地遍历网络中的所有层（卷积层、池化层、全连接层等）。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0.0)
