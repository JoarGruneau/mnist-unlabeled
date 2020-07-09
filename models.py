from torchvision import models

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torch import nn


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DecodeBlock, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channel,
                                          in_channel // 2,
                                          2,
                                          stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel // 2, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(True))

    def forward(self, x):
        out = self.deconv1(x)
        out = self.conv(out)
        return out


class Encoder(ResNet):
    def __init__(self, dims, name):
        self.name = name
        super(Encoder, self).__init__(BasicBlock, [2, 2, 2, 2],
                                      num_classes=dims)  # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
        self.conv1 = nn.Conv2d(1,
                               64,
                               kernel_size=7,
                               stride=1,
                               padding=3,
                               bias=False)
        self.activation = nn.Tanh()

    def forward(self, data):
        features = super(Encoder, self).forward(data)
        return self.activation(features)

    def get_name(self):
        return self.name


class Decoder(nn.Module):
    def __init__(self, in_channels, name):
        self.name = name
        super(Decoder, self).__init__()
        self.Linear_up = nn.Linear(in_channels, 64 * 4 * 4)

        self.deconvBlock1 = nn.Sequential(DecodeBlock(64, 64))
        self.deconvBlock2 = nn.Sequential(DecodeBlock(64, 32))
        self.deconvBlock3 = nn.Sequential(DecodeBlock(32, 16))
        self.conv1 = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x = self.Linear_up(x)
        x = x.view(-1, 64, 4, 4)
        x = self.deconvBlock1(x)
        x = self.deconvBlock2(x)
        x = self.deconvBlock3(x)
        x = self.conv1(x)
        return x

    def get_name(self):
        return self.name
