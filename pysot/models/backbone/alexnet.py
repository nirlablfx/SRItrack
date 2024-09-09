from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torchvision.models as models

# import torch, gc
#
# gc.collect()
# torch.cuda.empty_cache()

alexnet_offical = models.alexnet(pretrained = True)


class AlexNetLegacy(nn.Module):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), AlexNet.configs))
        super(AlexNetLegacy, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.features(x)
        return x


class AlexNet(nn.Module):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1,used_layers=[2,3,4]):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), AlexNet.configs))
        super(AlexNet, self).__init__()
        self.used_layers = used_layers

        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11,stride=2),#
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5,padding=1),#
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),#
            nn.ReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3,padding=1),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=3,padding=1),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=3,padding=1),
            nn.BatchNorm2d(configs[5]),
            )
        self.feature_size = configs[5]

    def forward(self, x):
        x1 = self.layer1(x)  #x 12 3 127 127 ,x1 12 96 29 29     255* 255 61 61 
        x2 = self.layer2(x1) #12 256 14 14     30*30
        x3 = self.layer3(x2) #12 384 14 14       26 26

        x4 = self.layer4(x3) #12 384 8 8
        x5 = self.layer5(x4) #12 256 6 6
        out = [x1,x2,x3,x4,x5]
        out = [out[i] for i in self.used_layers]
        return out


def alexnetlegacy(**kwargs):
    return AlexNetLegacy(**kwargs)


def alexnet(**kwargs):
    return AlexNet(**kwargs) 

