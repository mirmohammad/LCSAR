import torch
import torch.nn as nn


class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()

        self.convs = nn.Sequential(
            # conv1_1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # conv1_2
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # pool1
            nn.MaxPool2d()

        )

    def forward(self, x):
        pass
