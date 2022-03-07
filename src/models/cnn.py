import math

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features,
                               32,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.m1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(64,
                               512,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.act = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(4, 4))
        self.bn1=nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(512)
        self.feature=nn.Sequential(self.conv1,self.conv2,self.conv3)
        #self._initialize_weights()
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)
        x = torch.flatten(x, 1)
        x = self.act(self.classifier1(x))
        x = self.fc2(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()