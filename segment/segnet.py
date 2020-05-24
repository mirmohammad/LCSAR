import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, num_classes=21):
        super(SegNet, self).__init__()

        blocks = [2, 2, 3, 3, 3]
        kernel = [1, 64, 128, 256, 512, 512]

        for i in range(len(blocks)):
            block = []
            for j in range(blocks[i]):
                z = i if j == 0 else i + 1
                block.append(nn.Conv2d(kernel[z], kernel[i + 1], kernel_size=3, padding=1))
                block.append(nn.BatchNorm2d(kernel[i + 1]))
                block.append(nn.ReLU(inplace=True))
            block.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            setattr(self, 'encode' + str(i + 1), nn.Sequential(*block))

        blocks = [3, 3, 2, 2, 2]
        kernel = [512, 512, 256, 128, 64, num_classes]

        for i in range(len(blocks)):
            block = []
            for j in range(blocks[i]):
                z = i if j != blocks[i] - 1 else i + 1
                block.append(nn.Conv2d(kernel[i], kernel[z], kernel_size=3, padding=1))
                block.append(nn.BatchNorm2d(kernel[z]))
                block.append(nn.ReLU(inplace=True))
            setattr(self, 'unpool' + str(len(blocks) - i), nn.MaxUnpool2d(kernel_size=2, stride=2))
            setattr(self, 'decode' + str(len(blocks) - i), nn.Sequential(*block))

    def forward(self, x):

        # Encoder
        x, idx1 = self.encode1(x)
        x, idx2 = self.encode2(x)
        x, idx3 = self.encode3(x)
        x, idx4 = self.encode4(x)
        x, idx5 = self.encode5(x)

        # Decoder
        x = self.unpool5(x, indices=idx5)
        x = self.decode5(x)
        x = self.unpool4(x, indices=idx4)
        x = self.decode4(x)
        x = self.unpool3(x, indices=idx3)
        x = self.decode3(x)
        x = self.unpool2(x, indices=idx2)
        x = self.decode2(x)
        x = self.unpool1(x, indices=idx1)
        x = self.decode1(x)

        return x
