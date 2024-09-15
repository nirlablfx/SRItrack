import torch
import torch.nn as nn



import torch.nn as nn
import torch.nn.functional as F




class SPRIBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPRIBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out


class SPRI(nn.Module):
    def __init__(self, num_blocks, num_features):
        super(SPRI, self).__init__()
        self.num_blocks = num_blocks
        self.num_features = num_features

        # SPRI blocks
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()


        for _ in range(self.num_blocks):
            self.down_blocks.append(nn.Sequential(
                SPRIBlock(self.num_features, self.num_features),
                nn.MaxPool2d(1, 1)#2,2
            ))
            self.up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                SPRIBlock(self.num_features, self.num_features)
            ))

    def forward(self, x):
        down_outputs = []

        # Downward pass
        for i in range(self.num_blocks):
            x = self.down_blocks[i](x)
            down_outputs.append(x)

        # Upward pass
        for i in range (2) :#self.num_blocks - 1 ):#-1, -1


            x = x + down_outputs[i]
            upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

            if i ==0:
             down_outputs[i+1] = upsample1(down_outputs[i+1])
            if i ==1:
                down_outputs[i] = upsample2(down_outputs[i])


            x = self.up_blocks[i](x)

            return x








