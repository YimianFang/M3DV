import torch
import torch.nn as nn

class DenseBlockLayer(nn.Module):
    def __init__(self, input_channel, filters=16, bottleneck=4):
        super(DenseBlockLayer, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm3d(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(input_channel, filters * bottleneck, kernel_size=1),
            nn.BatchNorm3d(filters * bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters * bottleneck, filters, kernel_size=3, padding=1)
            # nn.Dropout3d(0.1)
        )

    def forward(self, x):
        x1 = self.net(x)
        x = torch.cat((x, x1), dim=1)
        return x


class DenseSharp(nn.Module):
    def __init__(self):
        super(DenseSharp, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            # nn.Dropout3d(0.1)
        )
        self.denseblock1 = nn.ModuleList([DenseBlockLayer(32 + i * 16) for i in range(4)])
        self.trans1 = nn.Sequential(
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 48, kernel_size=1),
            nn.AvgPool3d(kernel_size=2),
            # nn.Dropout3d(0.2)
        )
        self.denseblock2 = nn.ModuleList([DenseBlockLayer(48 + i * 16) for i in range(4)])
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(112),
            nn.ReLU(inplace=True),
            nn.Conv3d(112, 56, kernel_size=1),
            nn.AvgPool3d(kernel_size=2),
            # nn.Dropout3d(0.2)
        )
        self.denseblock3 = nn.ModuleList([DenseBlockLayer(56 + i * 16) for i in range(4)])
        self.trans3 = nn.Sequential(
            nn.BatchNorm3d(120),
            nn.ReLU(inplace=True),
            nn.Conv3d(120, 60, kernel_size=1),
            nn.AvgPool3d(kernel_size=2),
        )
        self.denseblock4 = nn.ModuleList([DenseBlockLayer(60 + i * 16) for i in range(4)])
        self.trans4 = nn.Sequential(
            nn.BatchNorm3d(124),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=4)
        )
        self.output = nn.Sequential(
            nn.Dropout(0.81),
            # nn.Linear(60*4*4*4, 512),
            # nn.Linear(512, 2),
            # nn.Softmax(dim=1)
            nn.Linear(124, 1),
            # nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        for i in range(4):
            x = self.denseblock1[i](x)
        x = self.trans1(x)
        for i in range(4):
            x = self.denseblock2[i](x)
        x = self.trans2(x)
        for i in range(4):
            x = self.denseblock3[i](x)
        x = self.trans3(x)
        for i in range(4):
            x = self.denseblock4[i](x)
        x = self.trans4(x)
        x = x.view(x.shape[0], -1)
        # x = self.output(x)
        x = self.output(x).squeeze()
        return x