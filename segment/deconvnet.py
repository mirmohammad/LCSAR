import torch.nn as nn


class DeconvNet(nn.Module):
    def __init__(self, num_classes=21):
        super(DeconvNet, self).__init__()

        self.conv1 = nn.Sequential(
            # conv1_1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # conv1_2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv2 = nn.Sequential(
            # conv2_1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # conv2_2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv3 = nn.Sequential(
            # conv3_1
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv3_2
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv3_3
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv4 = nn.Sequential(
            # conv4_1
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv4_2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv4_3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv5 = nn.Sequential(
            # conv5_1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv5_2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv5_3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.fc = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            # fc7
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
        )

        self.deconv6 = nn.Sequential(
            # deconv6
            nn.ConvTranspose2d(4096, 512, kernel_size=7),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconv5 = nn.Sequential(
            # deconv5_1
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # deconv5_2
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # deconv5_3
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconv4 = nn.Sequential(
            # deconv4_1
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # deconv4_2
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # deconv4_3
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconv3 = nn.Sequential(
            # deconv3_1
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # deconv3_2
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # deconv3_3
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconv2 = nn.Sequential(
            # deconv2_1
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # deconv2_2
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.deconv1 = nn.Sequential(
            # deconv1_1
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # deconv2_2
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.score = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):

        # convs
        x = self.conv1(x)
        x, idx1 = self.pool1(x)
        x = self.conv2(x)
        x, idx2 = self.pool2(x)
        x = self.conv3(x)
        x, idx3 = self.pool3(x)
        x = self.conv4(x)
        x, idx4 = self.pool4(x)
        x = self.conv5(x)
        x, idx5 = self.pool5(x)

        # fc
        x = self.fc(x)

        # deconvs
        x = self.deconv6(x)
        x = self.unpool5(x, indices=idx5)
        x = self.deconv5(x)

        x = self.unpool4(x, indices=idx4)
        x = self.deconv4(x)
        x = self.unpool3(x, indices=idx3)
        x = self.deconv3(x)
        x = self.unpool2(x, indices=idx2)
        x = self.deconv2(x)
        x = self.unpool1(x, indices=idx1)
        x = self.deconv1(x)

        # score
        x = self.score(x)

        return x
